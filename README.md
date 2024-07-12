Sequence-to-Sequence (Seq2Seq) attention model for Natural Language Processing (NLP) with PyTorch, based on [this tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html). Added configurable layers for the Gated Recurrent Unit (GRU) cells, and [FastText word embeddings](https://fasttext.cc/docs/en/crawl-vectors.html).

- Attention mechanism resulted in noticably better translations (comparison not shown but tested).
- Slightly better translations observed with than without pre-trained word embeddings (see comparison below).

Three implementations of word embeddings were compared. First was without any pre-trained embeddings, meaning all weights in the encoder and decoder embedding layers were randomly initialised. Second was with frozen pre-trained weights. Third was with non-frozen pre-trained weights (i.e. used only for initialisation). Of note, less than 1% of the corpus (0.65% for French, 0.15% for English) had to be initialised randomly because those words were not found in the FastText embeddings.

![image](https://github.com/user-attachments/assets/8a50f579-f144-43c2-a3e0-963b29a4f5b1)

<div align="center">
	<img src="https://github.com/user-attachments/assets/5f79fefd-6a4f-4188-99c2-d0c2bca498fa">
</div>

![image](https://github.com/user-attachments/assets/ad973666-048b-4cea-94a0-2fb0ea01897f)
