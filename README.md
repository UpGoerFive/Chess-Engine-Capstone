# A Neural Network Chess Engine

A project to build a chess engine using Neural Networks as evaluation metrics. The best performing CNN model achieved 90% accuracy predicting whether a move was good or bad. The models were trained on puzzle data from [Lichess](https://lichess.org/).

![Board Channels](https://github.com/UpGoerFive/Chess-Engine-Capstone/blob/ea5ac1f8248a2cc16fac47ef1895d7b5b1d4ea9f/images/board_breakdown.gif)

## Model Selection

Modern chess engines generally fall into two categories. More conventional engines consist of a searching algorithm that makes use of a handmade evaluation metric for the nodes, whereas deep learning engines, such as Alpha Zero or Leela Chess Zero, use reinforcement learning to train. This project is focused on applying neural networking techniques to the evaluation metric of the first type of engine.

Two things make using neural networks for evaluation metrics a viable application. The first is that the evaluation metric needs to return a value for the position based on the current state of the board alone; calculating moves ahead is what the search algorithm is for. The second is that evaluation metrics so far have tended to be manually built and tuned. Using neural networks in this context demonstrates how other evaluation problems could be similarly dealt with. I've focused on convolutional neural nets for this project, in the hope that the network could pay attention to positional features, though there is also a multilayer perceptron for comparison.

## Data

This project uses Lichess puzzles to train and evaluate on. The puzzle data can be downloaded [here](https://database.lichess.org/#puzzles). After unziping into the main repository directory, the `dataorg` notebook can be used to generate the train test split, and advance each FEN by one move (needed because of how Lichess hands off the data). The second half of that notebook produces data for early modeling that is not generally needed otherwise and will take a while to run. I've left it in for completeness and reproducibility of the early models.

The majority of the data preparation is devoted to converting from FEN positions, to an 8x8x13 array. Each channel represents one piece type of one color (with one channel for empty) with pieces for that channel represented as 1 and others as 0. This allows us to pass the data into a network similarly to how image data is often converted to an array first. It was necessary to create a custom data loader for this, which can be found in `data_generation.py` (for more information on how to do this, check out [this](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly) helpful blog post).

Based on how many positions the conversion produced, a single puzzle position has around 32 possible moves, and since these are "Only move" puzzles, only 1 of those moves is labeled as correct. This produces a large imbalance, and is the reason why the early models score around 96% accuracy on the validation data, despite being quite terrible. Guessing "bad move" for every single entry would produce around 97% accuracy.

To correct for this, later models only train on at most 4 bad positions in addition to the correct one (1 in 5 positions in the data are correct moves). This also reduced the the computation needed to convert from FENs for all training moves, making it possible to use the full dataset in training and evaluation, which was not feasible for me before.

## Model Selection

Almost all models in this project are Convolutional Neural Networks. The idea behind this was to capture features such as center control and connected pawns using the filtering that's part of a CNN, though obviously a neural network likely isn't capturing these things directly. For comparison I've included a multilayer perceptron, which performed just slightly worse than the starting CNN, with over ten times as many parameters. Using [Keras Tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) produced a CNN model that as expected outperformed all previous models. Interestingly, the tuner selected a filter size of 8, in other words each filter looks at the whole board area.

Each model was set to train for 30 epochs, though some were early stopped before then. The tensorboard for this project is available [here](https://tensorboard.dev/experiment/Ay6PmptxTvKgr5QtaNecQQ/#scalars). Note that both starter models using all possible moves and the later models with a reduced imbalance and more data to train from will be displayed on the graphs by default, so it is recommended to deliberately select which runs to compare. Both the [`full-puzzle-model`](https://github.com/UpGoerFive/Chess-Engine-Capstone/blob/cb455fd43c5f7d84ea5874fb2ac5780aaff5f537/full-puzzle-model.ipynb) and [`tuning`](https://github.com/UpGoerFive/Chess-Engine-Capstone/blob/da141f30f1fecf134dac62cbf00977b6fb042174/tuning.ipynb) notebooks can be run after the data preparation discussed above, though each individual model takes about a day to run on my machine.

### Searching

Included in `player.py` is a `Searcher` class, that implements [alpha-beta-pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning) for use in local play using the `Player` and `SingleGame` classes. Currently it executes too slowly to be effectively used in the version of the bot that's up on Lichess, but it still demonstrates how these types of networks can be used in engines.

## Performance

As stated above, early models achieved 96% accuracy, which while worse than a dumb model, does at least indicate there is something the network is picking up on which causes it to guess a position is correct. The model that trained on less imbalanced and more data achieved around 90% accuracy, as compared to an accuracy of around 80% if a model predicted "bad move" for every entry. Both engines do not perform particularly well in actual play, but this is expected as they are on their own without a searching algorithm.

## Play on Lichess

This project uses [Python-Chess](https://github.com/niklasf/python-chess) for all [UCI](https://backscattering.de/chess/uci/) functions. This includes generating legal moves, and getting board FENs. To enable play on Lichess, I've forked the [`lichess-bot`](https://github.com/UpGoerFive/lichess-bot) repository as recommended. The bot can be run on lichess if desired with the instructions provided in the README. This includes going through the setup to create an environment and connect an account, as your bot will need its own Lichess Oauth token. Models can be run locally for play using the `rolloutboard` notebook. Available models to choose from are in the Models directory. To select a different model to use in play, the model must be present in the `lichess-bot` repository, and the model path in the `PuzzledBot` class in `strategies.py` needs to be changed.

If you'd like to play the most recent version of the bot, you can find it [here](https://lichess.org/@/PuzzledBot) on Lichess.

## Resources

Here is a list of resources used while making this project:

- As to be expected, the documentation for Python, Numpy, Pandas, Tensorflow, and Keras
- [Keras Tuner Intro](https://keras.io/guides/keras_tuner/getting_started/)
- [Tensorboard Intro](https://www.tensorflow.org/tensorboard/get_started)
- [Python-Chess Documentation](https://python-chess.readthedocs.io/en/latest/)
- [Alpha Beta Pruning](https://www.chessprogramming.org/Alpha-Beta) on the chess programming wiki and [IDDFS](https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search)
- [Pillow Documentation](https://pillow.readthedocs.io/en/stable/) for the presentation images.

## Thanks!

Presentation slides can be found [here](https://www.canva.com/design/DAE54-TerF0/36Msz9xX1MZduSXrUSDRWw/view?utm_content=DAE54-TerF0&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink).

If you have questions or suggestions, please let me know!

**Repository Structure:**
```
Chess-Engine-Capstone
 ┣ environments
 ┃ ┣ Chess-Capstone.yml                                     <- Environment for running the models
 ┃ ┗ chess-plotting.yml                                     <- Environment for presentation images
 ┣ fens                                                     <- This directory and its files will be created after running dataorg
 ┃ ┣ test.csv
 ┃ ┣ train.csv
 ┃ ┗ val.csv
 ┣ images                                                   <- Presentation images
 ┣ misc-notebooks                                           <- Older and smaller notebooks
 ┃ ┣ README.md
 ┃ ┣ baseline-models.ipynb                                  <- Contains modeling code for very imbalanced, small models
 ┃ ┣ fenpreprocessing.py
 ┃ ┣ presentation-images.ipynb                              <- Creates presentation images
 ┃ ┣ rolloutboard.ipynb                                     <- Can be used to play models directly, or against each other
 ┃ ┗ test_fenprep.py
 ┣ models                                                   <- Directory of model files to load into keras
 ┃ ┣ ImbalancedModel.h5
 ┃ ┣ MLPmodel-Long.h5
 ┃ ┣ MLPmodel.h5
 ┃ ┣ ShortTrainImbalanced.h5
 ┃ ┣ full-puzzle.h5
 ┃ ┗ tuned_model.h5
 ┣ pdfs
 ┣ .gitignore
 ┣ README.md
 ┣ data_generation.py                                       <- Preprocessing and generator code
 ┣ dataorg.ipynb                                            <- Executes preprocessing
 ┣ full-puzzle-model.ipynb                                  <- Runs first full data CNN
 ┣ lichess_db_puzzle.csv                                    <- This needs to be downloaded from the above listed site.
 ┣ player.py                                                <- Engine code and local play ability
 ┗ tuning.ipynb                                             <- MLP model and Tuned CNN
 ```
