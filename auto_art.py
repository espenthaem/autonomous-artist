import config
import datetime
import numpy as np
import random
from numpy.random import choice, randint
from PIL import Image
import sys
sys.path.append(config.path_to_style_transfer)
sys.path.append(config.path_to_style_transfer + '/src')
from evaluate import simple_evaluate


class AutoArt:
    def __init__(self, museum='general', image_source='examples/content/chicago.jpg'):
        self.date = datetime.datetime
        self.image_source = image_source
        self.museum = museum
        self.style_models = config.trained_models[museum]
        self.image = np.array(Image.open(self.image_source).convert('RGB'))
        self.random_styled_images = []
        self.random_models = None
        self.final_image = self.image.copy()
        self.shape = self.image.shape
        self.strips = [None, None]
        self.gif_frames = []
        self.frame_duration = []

    def reset(self):
        self.final_image = self.image.copy()
        self.strips = [None, None]
        self.gif_frames = []

    def paint(self):
        self.random_models = np.random.choice(self.style_models, size=3, replace=False)
        for random_model in self.random_models[-3:]:
            model_path = 'checkpoint/' + self.museum + '/' + random_model
            print(model_path)
            self.random_styled_images.append(simple_evaluate(self.final_image, model_path))

    def _repaint(self, iterations=2):
        self.reset()
        repaintings = []
        for i in range(iterations):
            self.paint()
            # Clear strip history to assert unique stripping pattern per iteration
            self.strips = [None, None]
            # Increase number of strips with each iteration
            self.stitch_paintings(strips=randint(10, 11), axis=0)
            self.stitch_paintings(strips=randint(10, 11), axis=1)
            repaintings.append(self.final_image.copy())

        return repaintings

    def stitch_paintings(self, strips=10, axis=1, save_frames=False):
        # Save current number of frames, to be used for calculation of the frame duration
        previous_length = len(self.gif_frames)

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
                    self.gif_frames.append(Image.fromarray(self.final_image))
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
                        self.gif_frames.append(Image.fromarray(self.final_image))

        if save_frames:
            # Calculate frame duration, so every stitching procedure takes an equal amount of time the resulting GIF
            length_difference = len(self.gif_frames) - previous_length
            self.frame_duration = self.frame_duration + list(np.ones(length_difference)/length_difference)

    def _create_stitching_frames(self, num_frames=10):
        # Protected for aesthetic reasons
        # For aesthetic reasons, create first frame during first stitching iteration on a random axis
        random_axis = choice(2)
        self.stitch_paintings(axis=random_axis, strips=randint(1, 20), save_frames=True)

        # Stitch the other axis without saving the intermediate frames
        self.stitch_paintings(axis=(1 - random_axis), strips=randint(1, 20), save_frames=True)

        # Generate remaining frames
        for i in range(num_frames - 1):
            print('Creating frame # %s' % i)
            # Perform random stitching and save to frames array as PIL Image
            self.stitch_paintings(axis=choice(2), strips=randint(1, 20))
            self.gif_frames.append(Image.fromarray(self.final_image))

    def create_repaint_frames(self, iterations=2, intermediate_frames=10):
        # Construct array of images that will appear in the final GIF
        images = [self.image]
        iteration_images = self._repaint(iterations)
        for i in range(iterations):
            images.extend(self.random_styled_images[i * 3: i * 3 + 3])
            images.append(iteration_images[i])

        # Construct individual GIF frames of the blending of the actual images
        for i in range(len(images) - 1):
            blend_0 = Image.fromarray(images[i])
            blend_1 = Image.fromarray(images[i + 1])
            for j in np.linspace(0, 1, intermediate_frames):
                blended = Image.blend(blend_0, blend_1, j)
                self.gif_frames.append(blended)

    def create_gif(self, gif_path=None, reverse_frames=False, duration=250):
        """
        Creates a gif of the gif_frames contained int he class. The gif is saved to the same locations as
        the image source, unless another path is specified. Result are not very aesthetically pleasing, so
        this function is just left for visualisation of the process.
        :param gif_path:
        :param reverse_frames:
        :param duration:
        """

        assert len(self.gif_frames) > 0

        if not gif_path:
            gif_path = self.image_source[:self.image_source.rfind('.')] + '.gif'

        # Add reversed frames if required
        if reverse_frames:
            self.gif_frames = self.gif_frames + list(reversed(self.gif_frames[1:]))
            self.frame_duration = self.frame_duration + list(reversed(self.frame_duration[1:]))

        # Save frames as gif
        print(gif_path)
        self.gif_frames[0].save(gif_path,
                                format='GIF',
                                append_images=self.gif_frames[1:],
                                save_all=True,
                                duration=duration,
                                loop=0)


artist = AutoArt(museum='stedelijk', image_source='PATH/TO/example.jpg')

artist.create_repaint_frames(iterations=3, intermediate_frames=50)

artist.create_gif(duration=40, reverse_frames=True)
