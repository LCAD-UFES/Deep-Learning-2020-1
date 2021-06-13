
## [Estimativa de Profundidade de Alta Qualidade utilizando Camera Monocular via "Adaptive Bins"](https://github.com/LCAD-UFES/Deep-Learning-2020-1/blob/main/MapaDeProfundidade/relatorio/DepthMapUsingAdabins.pdf)
* por Aureziano Faria de Oliveira e Thiago Goncalves Cavalcante

## Modelos Pré-Treinados
Os modelos pré-treinados com "NYU depth v2" e "kitti" estão disponíveis [aqui](https://1drv.ms/u/s!AuWRnPR26byUmfRxBQ327hc8eXse2Q?e=AQuYZw)

## Preparando o Ambiente
* Para utilizar é necessário ter o CUDA_10.0 (e Cudnn compatível) , Python 3.5 (ou superior), pip e virtualenv (para evitar problemas)
* Certifique-se de baixar as dependencias no [link](https://1drv.ms/u/s!AuWRnPR26byUmfRbqEF7468fDdHM1g?e=KoabLc)
* Salve-as na pasta do projeto!
* Execute o comando e todas as dependências serão instaladas (processo já testado):
```
./instalar_dependencias.sh 1
```
O parametro 1 é para instalar CUDA e CUNN corretamente, se você já possuir o CUDA 10.0 e Cudnn compatível instalados então rode o seguinte comando:
```
./instalar_dependencias.sh
```

## Treinar (Precisa de uma Titan V)
* Para treinar você precisará de um login no wandb. Crie suas credenciais no site do [wandb](https://wandb.ai/site) e as memorize para utilização.
* Realize o login no terminal e inicie o treino:
```
   source venv/bin/activate
   wandb login
   python3 train.py --dataset nyu --gpus 1 --bs 4
   deactivate
```

## Testando a rede

## Visualizar imagens lado a lado - original/profundidade
* Rode o seguinte comando para utilizar um vídeo e gerar imagens lado a lado comparando o frame original e a profundidade estimada:
```
python infer_video.py --model kitti --input test_video.mp4
```
Obs.: test_video.mp4 é algum vídeo de sua escolha.

#Videos
* [Teste na IARA](https://drive.google.com/file/d/1Okb38k_hgC2Um9GGgPdQYeZEGTztFQrP/view?usp=sharing)
* [Teste no ART](https://drive.google.com/file/d/1l__YR7KJaUGiOFkxkbPSHsNXn7Q4iSlk/view?usp=sharing)
* [Reconstrução 3D de cena](https://drive.google.com/file/d/1oTzFkOW8UthiehopQ3-HCS9LUFdFTdek/view?usp=sharing)

# Nossas contribuições
* View3dPointCloud.py: Script python para visualizar a reconstrução 3D a partir das imagens do ART (calibrado para cameras intelbras)
* DeepMapper.ipynb: Script notebook para executar diretamente do Google Colab
* infer_video.py: Script python para gerar video comparativo das imagens RGB e de Profundidade
* Modulo deep_mapper do carmen: Integração com o sistema autônomo utilizado na IARA e no ART, disponível [aqui](https://github.com/LCAD-UFES/carmen_lcad/tree/master/src/deep_mapper/)
* instalar_dependencias.sh: Script shell para instalação do virtualenv e dependencias, evitando conflitos com versões pré-existentes.
* raw_data_downloader.sh: Adaptação do Script shell original para download do dataset kitti que pode ser interrompido e reinicia de onde parou.

# Artigo original
[AdaBins](https://arxiv.org/abs/2011.14141)
