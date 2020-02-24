import config
import datetime
import numpy as np
import random
import sys
import cv2 as cv
from PIL import Image
from numpy.random import choice, randint
sys.path.append(config.path_to_style_transfer)
sys.path.append(config.path_to_style_transfer + '/src')
from evaluate import simple_evaluate
from utilities import smallest_divisor
from utilities import alpha_blend
from perlin_noise import generate_perlin_noise_2d, generate_perlin_noise_3d


class AutoArt:
    def __init__(self, museum='general', image_source='examples/content/exactitudes.jpg', size=None):
        self.date = datetime.datetime
        self.image_source = image_source
        self.museum = museum
        self.style_models = config.trained_models[museum]

        if size is None:
            self.image = np.array(Image.open(self.image_source).convert('RGB'))
        else:
            self.image = np.array(Image.open(self.image_source).convert('RGB').resize(size))

        self.random_styled_images = []
        self.random_models = None
        self.final_image = self.image.copy()
        self.shape = self.image.shape
        self.strips = [None, None]
        self.frames = []

    def reset(self):
        self.final_image = self.image.copy()
        self.strips = [None, None]
        self.frames = []

    def paint(self):
        self.random_models = np.random.choice(self.style_models, size=3, replace=False)
        for random_model in self.random_models[-3:]:
            model_path = 'checkpoint/' + self.museum + '/' + random_model
            print(model_path)
            self.random_styled_images.append(simple_evaluate(self.final_image, model_path))

    def _repaint(self, type='stitch'):
        assert type in ['stitch', 'perlin']
        self.paint()
        # Blend style images using appropriate mechanism
        if type == 'stitch':
            # Clear strip history to assert unique stripping pattern per iteration
            self.strips = [None, None]
            # Increase number of strips with each iteration
            self.stitch_paintings(strips=randint(10, 11), axis=0)
            self.stitch_paintings(strips=randint(10, 11), axis=1)
        elif type == 'perlin':
            self.perlin_blend_paintings()

        return self.final_image.copy()

    def stitch_paintings(self, strips=10, axis=1, save_frames=False):
        # Save current number of frames, to be used for calculation of the frame duration
        previous_length = len(self.frames)

        # Start Stitching procedure
        self.strips[axis] = strips
        strip_size = int(self.shape[axis] / strips)

        # Check if other axis strips are present
        if not self.strips[1-axis]:
            for strip in range(strips):
                style_img = random.choice(self.random_styled_images[-3:])
                if axis == 1:
                    self.final_image[:, strip*strip_size:(strip+1)*strip_size] = \
                        style_img[:, strip*strip_size:(strip+1)*strip_size]
                elif axis == 0:
                    self.final_image[strip * strip_size:(strip + 1) * strip_size, :] = \
                        style_img[strip * strip_size:(strip + 1) * strip_size, :]
                if save_frames:
                    self.frames.append(Image.fromarray(self.final_image))
        else:
            o_strips = self.strips[1-axis]
            o_size = int(self.shape[axis]/strips)

            # Pick random other strips to apply current strip to
            for o_strip in choice(o_strips, int(o_strips/2)):
                style_img = random.choice(self.random_styled_images[-3:])

                for strip in choice(strips, int(strips)):
                    if axis == 1:
                        self.final_image[o_strip*o_size:(o_strip+1)*o_size, strip * strip_size:(strip + 1) * strip_size] = \
                            style_img[o_strip*o_size:(o_strip+1)*o_size, strip * strip_size:(strip + 1) * strip_size]
                    elif axis == 0:
                        self.final_image[strip * strip_size:(strip + 1) * strip_size, o_strip*o_size:(o_strip+1)*o_size] = \
                            style_img[strip * strip_size:(strip + 1) * strip_size, o_strip*o_size:(o_strip+1)*o_size]
                    if save_frames:
                        self.frames.append(Image.fromarray(self.final_image))

    def perlin_blend_paintings(self, noise=None):
        if noise is None:
            # Determine the amount of periods of noise to generate along each axis
            res_x = smallest_divisor(self.shape[0])
            res_y = smallest_divisor(self.shape[1])
            noise = generate_perlin_noise_2d(self.shape[:2], (res_x, res_y))
        assert noise.max() == 1 and noise.min() == 0
        # Resize noise level to match the amount of randomized style models
        noise = noise * 3
        # Create map
        map = noise.repeat(3, axis=1).reshape((self.shape[0], self.shape[1], 3))

        # Blend 3 random styled images using a circle representation of the noise level
        # 0 -> A 1 -> B 2 -> C 3 -> A
        map[map == 3] = 0
        region_0 = (0 <= map) & (map < 1)
        region_1 = (1 <= map) & (map < 2)
        region_2 = (2 <= map) & (map < 3)

        A = self.random_styled_images[-3]
        B = self.random_styled_images[-2]
        C = self.random_styled_images[-1]

        blended_image = np.zeros(map.shape)

        # Blend regions
        blended_image[region_0] = A[region_0] * (1 - map[region_0]) + B[region_0] * map[region_0]

        blended_image[region_1] = B[region_1] * ((1 - map[region_1]) % 1) + C[region_1] * (map[region_1] % 1)

        blended_image[region_2] = C[region_2] * ((1 - map[region_2]) % 1) + A[region_2] * (map[region_2] % 1)

        self.final_image = blended_image.astype('uint8').copy()

    def create_stitch_frames(self, iterations=2, intermediate_frames=10):
        # Construct array of images that will appear in the final GIF
        images = [self.image]
        for i in range(iterations):
            repainting = self._repaint(type='stitch')
            images.extend(self.random_styled_images[-3:])
            images.append(repainting)

        # Construct individual GIF frames using simple alpha blending
        for i in range(len(images) - 1):
            blend_0 = Image.fromarray(images[i])
            blend_1 = Image.fromarray(images[i + 1])
            self.frames.extend(alpha_blend(blend_0, blend_1, intermediate_frames))

    def create_perlin_frames(self, iterations=2, intermediate_frames=10):
        # Generate 3D perlin noise to blend iterations of styled images
        res_x = smallest_divisor(self.shape[0])
        res_y = smallest_divisor(self.shape[1])
        res_z = smallest_divisor(intermediate_frames)
        noise_3d = generate_perlin_noise_3d((self.shape[0], self.shape[1], intermediate_frames),
                                            (res_x, res_y, res_z))
        # Start with the actual input image
        self.frames.append(Image.fromarray(self.image))

        for i in range(iterations):
            repainting = self._repaint(type='perlin')
            # Create intermediate frames by applying a 2D slice of 3D blending noise to the styled images
            for j in range(intermediate_frames):
                # Scale 2D slice of 3D noise to [-1, 1]
                noise_2d = noise_3d[:, :, j]
                noise_2d = (noise_2d - noise_2d.min()) / (noise_2d.max() - noise_2d.min())
                self.perlin_blend_paintings(noise=noise_2d)
                # Transition from the last frame of the previous iteration to first intermediate frame
                # using alpha blending
                if j == 0:
                    blend_0 = self.frames[-1]
                    blend_1 = Image.fromarray(self.final_image)
                    self.frames.extend(alpha_blend(blend_0, blend_1, int(intermediate_frames/4)))
                else:
                    self.frames.append(Image.fromarray(self.final_image))

            # Transition from the final blend frame to the repainting using simple alpha blending
            blend_0 = self.frames[-1]
            blend_1 = Image.fromarray(repainting)
            self.frames.extend(alpha_blend(blend_0, blend_1, int(intermediate_frames/4)))

    def create_animation(self, out_path, reverse_frames=False, fps=20):
        """
        Creates an animation of the frames contained in the class. The animation is saved as either .gif or .avi,
        depending on the out_path that is specified.

        :param out_path: str
        :param reverse_frames: bool
        :param fps: int
        """

        assert len(self.frames) > 0
        assert 'gif' in out_path or 'avi' in out_path

        # Add reversed frames if required
        if reverse_frames:
            frames = self.frames + list(reversed(self.frames[1:]))
        else:
            frames = self.frames

        print(out_path)
        if 'gif' in out_path:
            # Save frames as gif
            frames[0].save(out_path,
                           format='GIF',
                           append_images=self.frames[1:],
                           save_all=True,
                           duration=1000/fps,
                           loop=0)

        elif 'avi' in out_path:
            # Save frames as video
            four_cc = cv.VideoWriter_fourcc(*'DIVX')
            writer = cv.VideoWriter(out_path, four_cc, fps, (self.shape[1], self.shape[0]))
            for frame in frames:
                writer.write(cv.cvtColor(np.array(frame), cv.COLOR_RGB2BGR))
            writer.release()


if __name__ == '__main__':
    artist = AutoArt(museum='stedelijk', image_source='examples/content/exactitudes.jpg')

    artist.create_perlin_frames(iterations=3, intermediate_frames=50)

    artist.create_animation(out_path='examples/results/perlin_exactitudes.avi', reverse_frames=True, fps=10)
