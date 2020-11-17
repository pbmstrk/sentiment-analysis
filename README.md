<div align="center">

<h1> Text Classification </h1>

*Implementations of models for text classification from various research papers*

</div>


The following models are implemented:
- [Convolutional Neural Networks for Sentence Classification (Kim, 2014)](https://www.aclweb.org/anthology/D14-1181/)
- [Neural Semantic Encoders (Munkhdalai and Yu, 2017)](https://arxiv.org/abs/1607.04315)
- Transformer Encoder - from [Attention is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Convolutional Neural Networks with Recurrent Neural Filters (Yang, 2018)](https://arxiv.org/abs/1808.09315)

Scripts to run models on SST dataset are provided.

## TextCNN

<p align="center">
  <img src="img/textcnn.png" width=625px/>
  <br>
  <em>TextCNN architecture. Source: (Kim, 2014)</em>
</p>

Run model using,

```
python scripts/textcnn/main.py
```

## Neural Semantic Encoders


<p align="center">
  <img src="img/nse.png" width=450px/>
  <br>
  <em>NSE architecture. Source: (Munkhdalai and Yu, 2017)</em>
</p>

Run model using,

```
python scripts/nse/main.py
```

## Transformer

[Attention is All You Need](https://arxiv.org/abs/1706.03762)

<p align="center">
  <img src="img/transformer.png" height=300px/>
  <br>
  <em>Transformer Encoder architecture. Source: (Vaswani et al., 2017)</em>
</p>

Run model using,

```
python scripts/transformer/main.py
```

## CNNs with Recurrent Neural Filters

Run model using,

```
python scripts/rnf/main.py
```
