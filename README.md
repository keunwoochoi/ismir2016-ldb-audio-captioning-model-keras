# ismir2016-ldb-audio-captioning-model-keras
Audio captioning model in Keras

### What does it do
This is a general sequence-to-sequence model. 

Details are in [my ISMIR 2016 LDB extended abstract](https://github.com/keunwoochoi/ismir2016-ldb-audio-captioning-model-keras/blob/master/audio_captioning_1.1_final_submission.pdf)

### Usage
TODO

# More details

### Motivations
```python
if recommendation.type == playlist:
    if generation_method == automatic:
        raise("No descrption! Users don't understand what the playlists are about and confused. Consider generating descriptions so that music discovery can be easier! ")
```
E.g.,

 * Playfully, silly or sublime--this is the sound of Paul in love. (for [PaulMcCartney Ballads playlist, Apple music](https://itunes.apple.com/us/playlist/michael-jackson-love-songs/idpl.8058d87c60b647a7bc81185b9f59e4c2))
 * Just the right blend of chilled-out acoustic songs to work, relax, and dream to (for [Your Coffee Break, Spotify playlist](https://open.spotify.com/user/spotify_uk_/playlist/48910w3L1DNiqvMHbUfZyY))
 
### Backgrounds
#### Previously,

 * Tags prediction for track *[(Eck et al., 2008)](http://papers.nips.cc/paper/3370-automatic-generation-of-social-tags-for-music-recommendation)*
 * Tags for playlists *[(Fields et al., 2010)](http://research.gold.ac.uk/8793/)*
 * Visual avatars for tracks *[(Bogdanov et al, 2013)](https://scholar.google.co.kr/scholar?cluster=14015036933742269998&hl=en&as_sdt=0,33)*

#### Some techniques we can use
 * **RNNs** for sequence modelling
 * **Seq2Seq** which uses RNNs to model the relationships between two sequences, e.g., two sentences in different languges
 * **Word2vec** or anyother word-embedding methods
 * **ConvNet** Convolutional neural networks for various tasks including music

### The proposed structure
![structure alt text](https://github.com/keunwoochoi/ismir2016-ldb-audio-captioning-model-keras/blob/master/imgs/ismir2016-ldb-captioning-diagram1-for-web.png "structure")
 *  Input
  - A sequence of track features
  - A track feature:
    - Concat(audio_feature, text_embedding)
      - audio_feature: audio content feature
      - text_embedding: text summarisation of text data (metadata, lyrics, descriptions...) of the track
        - text summasation method: averaging word embeddings of every word
 * Output
  - Sequence of word embeddings
    - each word embedding represent each word of the description (e.g. if the ground-truth description is *Playfully, silly or sublime--this is the sound of Paul in love*, `word_embedding(playfully)`, `word_embedding(silly)`, `word_embedding(or)`,... )


### The results
I don't have a proper result and looking for dataset. Anybody help? 


