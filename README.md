# Autonomously Generated Art

## Background
Harnesses the power of [fast-style-transfer](https://github.com/lengstrom/fast-style-transfer), as implemented by [Logan Engstrom](https://github.com/lengstrom), in an attempt to create new art. Existing style models are recombined and iteratively used to explore the possibilities using this framework. 

Available style models are specified in the config, and are placed in the checkpoint folder:

```
trained_models = {'general': ['la_muse',
                              'minotaur',
                              'rembrandt',
                              'scream',
                              'udnie',
                              'van_gogh'],
                  'stedelijk': ['tablea_iii',
                                'an_englishman_in_moscow',
                                'landschaft_in_dangast',
                                'rabbit_for_dinner_sandro',
                                'autoportrait_aux_sept_doigts',
                                'contra_compisitie_v',
                                'formes_circulaires_soleil_lune',
                                'martha']
                  }
```

The style models are grouped on museum level. The autonomous artist operates on a set of trained models that are specific to a certain museum. 

### Image Blending

Styled images are obtained by applying a random selection of style models to input image. Styled images can be combined using two different methods; stichting and perlin blending. Stitching combines the images by choosing random rectangles and setting the pixel values to one of the style images. Perlin blending combines the pictures using simple alpha blending in combination with an image mask that is derived from 2D Perlin noise. 

### Repainting
 
The recombination of the styled images is fed back into a new subset of style models. These new styled are then again blended. This process is called repainting and can be applied iteratively until aesthetically pleasing results are achieved. This process is visualized using the generation of frames. 

## Usage and Output Example
This example was generated using the first image of the Exactitude series by [Ari Versluis and Ellie Uyttenbroek](https://exactitudes.com/). Style models of works displayed in the Stedelijk Museum in Amsterdam were used.    
```
from auto_art import AutoArt


artist = AutoArt(museum='stedelijk', image_source='examples/content/exactitudes.jpg')

artist.create_perlin_frames(iterations=3, intermediate_frames=50)
artist.create_animation(out_path='examples/results/perlin_exactitudes.avi', reverse_frames=True, fps=10)

artist.create_stitch_frames(iterations=3, intermediate_frames=50)
artist.create_animation(out_path='examples/results/stitch_exactitudes.avi', reverse_frames=True, fps=10)
```

#### Stitching
<img src="https://thumbs.gfycat.com/FrighteningFrigidGoshawk-size_restricted.gif" width="350" height="350">

The uncompressed version can be viewed [here](https://gfycat.com/frighteningfrigidgoshawk)

#### Perlin Blending
<img src="https://thumbs.gfycat.com/DeliciousThoroughKillerwhale-size_restricted.gif" width="350" height="350">

The uncompressed version can be viewed [here](https://gfycat.com/deliciousthoroughkillerwhale)
