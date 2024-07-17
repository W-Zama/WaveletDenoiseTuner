# WaveletDenoiseTuner
ウェーブレット変換を用いた関数のノイズ除去を視覚化したツールです．

## 使用画面
<img src="https://github.com/user-attachments/assets/43937e3d-e9c5-43a5-8b02-c8ca50b744b0" alt="使用画面" width=50%>

## 機能
- スライダーによる閾値の指定
- 複数の関数による視覚化
- ソフト閾値法とハード閾値法の選択
- RMSEのリアルタイム算出

## 実行方法
python3をインストールした後，

```bash
pip install -r requirements.txt
```

により依存関係をインストールしてください．その後，

```bash
python main.py
```
と入力し，main.pyを実行してください．
