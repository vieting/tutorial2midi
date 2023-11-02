# tutorial2midi
tutorial2midi is a tool to convert piano tutorials in the style of a synthesia video to midi files.
This is helpful to obtain sheet music.
Creating a pdf with sheet music from a midi file is easy using software like MuseScore.

## Installation
tutorial2midi is a regular python project.
Just clone this repository and create an environment with the required packages, e.g. by running
```
git clone git@github.com:vieting/tutorial2midi.git
cd tutorial2midi
pip install -r requirements.txt
```

## Usage
The main entry point is `main.py`.
To see the available command line options, run `python main.py -h`.
An example use case would be
```
python main.py /path/to/tutorial.mp4 --tempo 75 --quantization 8 --anacrusis 0.5
```

## Related projects
A number of related projects exist.
The most popular one is probably [video2midi](https://github.com/svsdval/video2midi).
This project also takes inspiration from [pianoroll2midi](https://github.com/mattstaib/pianoroll_to_midi).
Other related projects include [MIDI-Converter](https://github.com/41pha1/MIDI-Converter), [DeSynthesia](https://github.com/kevinlinxc/DeSynthesia/), [synthesiavideo2midi](https://github.com/devbridie/synthesiavideo2midi), more are listed [here](https://edvein-rin.github.io/synthesia-video-converter/comparison-of-existing-solutions/).
However, for all projects I either had difficulties installing all necessary dependencies (e.g. for video2midi) or the results were completely off for the examples that I tested (e.g. pianoroll2midi).