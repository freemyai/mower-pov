"""Evaluate a centered shutter on the absolute timeline and integrate linear Cycles images."""

from dataclasses import dataclass
from math import isfinite
from time import perf_counter


@dataclass(frozen=True)
class Exposure:
    milliseconds: float
    samples: int

    def __post_init__(self):
        if not isfinite(self.milliseconds) or self.milliseconds < 0:
            raise ValueError("shutter duration must be finite and nonnegative")
        minimum = 3 if self.milliseconds else 1
        if type(self.samples) is not int or self.samples < minimum or self.samples % 2 == 0:
            raise ValueError(f"shutter samples must be an odd integer of at least {minimum}")

    def sample_frames(self, frame, fps, duration_s):
        if self.milliseconds == 0:
            return (frame,)
        width = self.milliseconds * fps / 1000
        if width > 1:
            raise ValueError("shutter duration must not exceed the frame period")
        start = max(0, frame - width / 2)
        stop = min(duration_s * fps, frame + width / 2)
        step = (stop - start) / self.samples
        return tuple(start + (i + .5) * step for i in range(self.samples))


class BlenderShutter:
    def __init__(self, scene, work, *, png, linear):
        import bpy

        self.bpy = bpy
        self.scene = scene
        self.work = work
        self.png = png
        self.linear = linear
        self.layered = linear or not png
        image_format = scene.render.image_settings
        if self.layered:
            image_format.media_type = "MULTI_LAYER_IMAGE"
            image_format.file_format = "OPEN_EXR_MULTILAYER"
        else:
            image_format.file_format = "OPEN_EXR"
            scene.view_layers[0].cycles.denoising_store_passes = False
        image_format.color_mode = "RGBA"
        image_format.color_depth = "32"
        image_format.exr_codec = "ZIP"

        # The file compositor reads the completed EXRs. Its own render is just a
        # one-pixel empty scene, and the input images define the output extent.
        composite = bpy.data.scenes.new("Shutter integration")
        composite.render.engine = "CYCLES"
        composite.cycles.samples = 1
        composite.cycles.use_denoising = False
        composite.render.resolution_x = composite.render.resolution_y = 1
        composite.render.compositor_device = "CPU"
        composite.render.compositor_precision = "FULL"
        composite.render.dither_intensity = scene.render.dither_intensity
        # File Output includes enabled stamp metadata even with burn-in disabled.
        for prop in composite.render.bl_rna.properties:
            if prop.identifier.startswith("use_stamp_"):
                setattr(composite.render, prop.identifier, False)
        composite.display_settings.display_device = scene.display_settings.display_device
        composite.view_settings.view_transform = scene.view_settings.view_transform
        composite.view_settings.look = scene.view_settings.look
        composite.view_settings.exposure = scene.view_settings.exposure
        composite.view_settings.gamma = scene.view_settings.gamma
        camera = bpy.data.objects.new("Compositor camera", bpy.data.cameras.new("Compositor camera"))
        composite.collection.objects.link(camera)
        composite.camera = camera
        self.composite = composite
        self.tree = bpy.data.node_groups.new("Linear shutter integration", "CompositorNodeTree")
        composite.compositing_node_group = self.tree

    def _mean(self, sockets):
        result = sockets[0]
        for count, socket in enumerate(sockets[1:], start=2):
            mix = self.tree.nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            mix.inputs[0].default_value = 1 / count
            self.tree.links.new(result, mix.inputs[6])
            self.tree.links.new(socket, mix.inputs[7])
            result = mix.outputs[2]
        return result

    def _output(self, path, png, channels):
        file = self.tree.nodes.new("CompositorNodeOutputFile")
        file.directory = str(path.parent)
        file.file_name = path.stem
        file.save_as_render = png
        if png:
            file.format.media_type = "IMAGE"
            file.format.file_format = "PNG"
            file.format.color_mode = "RGB"
            file.format.color_depth = "8"
        else:
            file.format.file_format = "OPEN_EXR_MULTILAYER"
            file.format.color_depth = "16"
            file.format.exr_codec = "ZIP"
        for index, (name, socket) in enumerate(channels):
            kind = "FLOAT" if socket.type == "VALUE" else socket.type
            file.file_output_items.new(kind, name)
            self.tree.links.new(socket, file.inputs[index])

    def _compose(self, frame, images, destination, sidecar):
        nodes = []
        for image in images:
            node = self.tree.nodes.new("CompositorNodeImage")
            node.image = image
            nodes.append(node)
        combined = self._mean([node.outputs[0] for node in nodes])
        reference = nodes[len(nodes) // 2].outputs
        if self.png:
            self._output(destination, True, [("", combined)])
        else:
            channels = [("ViewLayer.Combined", combined)]
            for socket in reference:
                if socket.name == "Noisy Image":
                    noisy = self._mean([node.outputs[socket.name] for node in nodes])
                    channels.append(("ViewLayer.Noisy Image", noisy))
                elif socket.name not in {"Combined", "Alpha"}:
                    channels.append((f"ViewLayer.{socket.name}", socket))
            self._output(destination, False, channels)
        if self.linear:
            raw = self._mean([node.outputs["Noisy Image"] for node in nodes])
            self._output(sidecar, False, [
                ("RawBeauty", raw),
                ("Albedo", reference["Denoising Albedo"]),
                ("Normal", reference["Denoising Normal"]),
            ])
        self.composite.frame_set(frame)
        self.bpy.ops.render.render(scene=self.composite.name)

    def render(self, frame, sample_frames, apply, destination, sidecar=None):
        state_time, render_times = 0., []
        images, paths = [], []
        try:
            for i, sample_frame in enumerate(sample_frames):
                started = perf_counter()
                apply(sample_frame)
                ready = perf_counter()
                state_time += ready - started
                path = self.work / f"sample-{i:03d}.exr"
                paths.append(path)
                self.scene.render.filepath = str(path)
                self.bpy.ops.render.render(write_still=True)
                images.append(self.bpy.data.images.load(str(path)))
                render_times.append(perf_counter() - ready)
            started = perf_counter()
            self._compose(frame, images, destination, sidecar)
            compose_time = perf_counter() - started
        finally:
            self.tree.nodes.clear()
            for image in images:
                self.bpy.data.images.remove(image)
            for path in paths:
                path.unlink(missing_ok=True)
        return state_time, render_times, compose_time
