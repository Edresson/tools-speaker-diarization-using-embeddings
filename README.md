# tools-speaker-diarization-using-embeddings
Este repositorio possui algumas redes e suas respectivas topologias para speaker diarization (Reconhecimento de locutor),
possuimos os seguintes modelos:

Rna Tradicional (Scripts/Redes/rna-tradicional.py): se trata de uma simples rede neural artificial utilizada para distinção de locutores através de classes utilizando softmax na saida da rede.
Embeddings (Scripts/Redes/encoder1.py) : se trata de uma arquitetura mais elaborada onde é possivel apartir de uma amostra de 5 segundo cadastrada anteriormente reconhecer qualquer locutor, utilizando distância euclediana. Este modelo conseguiu acertar 90% das amostras de novos locutores, ou seja neste caso não é necessário ter que treinar a rede com o locutor é apenas necessário possuir uma amostra de audio do locutor cadastrado.

Para mais detalhes consulte nosso artigo: Em Breve

For more details see our article: Coming soon
