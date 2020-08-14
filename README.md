# DeepDream

Simple Deep Dream Code

### Test:

```Bash
python main.py 
```

Arguments:
```
--img_file:     path to input image; default="images/supermarket.jpg"
--iterations:   number of gradient ascent steps per octave; default=20
--at_layer:     layer at which we modify image to maximize outputs; default=27
--lr:           learning rate; default=0.01
--octave_scale: image scale between octaves; default=1.4
--num_octaves:  number of octaves; default=10
```
