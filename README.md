MCTest Dataset
========

Baseline models as well as more complex ones for doing question answering on the MCTest dataset.

Dependencies:
```
protobuf
numpy
pandas
nltk
```

Word embeddings can be used from a model file created by [word2vec](https://github.com/danielfrg/word2vec).

## Running baseline models


First, clone the repo and compile the protobuf:
```
git clone https://github.com/mcobzarenco/mctest.git 
cd mctest
protoc --python_out=. mctest.proto
```

To parse the raw data (dev + train combined), remove stopwords and save it as a length delimted protobuf flat file:
```
cat data/MCTest/mc160.dev.tsv data/MCTest/mc160.train.tsv | \
  ./parse.py --rm-stop data/stopwords.txt -o proto > train160-stop.words
```

Also create a file with the ground truth for dev + train:
```
cat data/MCTest/mc160.dev.ans data/MCTest/mc160.train.ans > train160.ans 
```

To run the sliding window with distance baseline:
```
./baseline.py --train train160-stop.words --truth train160.ans --distance

[model]
window_size = None
distance = True

[results]
All accuracy [400]: 0.5600
Single accuracy [185]: 0.5946
Multiple accuracy [215]: 0.5302
```

#### Word embeddings
First, [word2vec](https://github.com/danielfrg/word2vec) should be installed and a model file with embeddings created.
Say the model file is `mctest.vec.bin`, the following command will parse the raw data (dev + train combined), replace the words with their corresponding embedding and save that to disk:
```
cat data/MCTest/mc160.dev.tsv data/MCTest/mc160.train.tsv | \
  ./parse.py --model-file mctest.vec.bin --rm-punct -o proto > train160-punct-mctest.embed
```
To run the sliding window model over the embeddings:
```
./baseline-embed.py --train train160-punct-mctest.embed --truth train160.ans 

[model]
window_size = None

All accuracy [400]: 0.5775
Single accuracy [185]: 0.6108
Multiple accuracy [215]: 0.5488
```
