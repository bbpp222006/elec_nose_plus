# 基于ctc与attention两种机制的不定长时间序列检测算法
大致算法会在这两者之间做选择.现在仍在查询资料和学习  
学习路线:https://github.com/zzw922cn/awesome-speech-recognition-speech-synthesis-papers
## ctc篇
[算法简介](http://xiaofengshi.com/2019/02/14/CTC%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86/)  
其实也只是看懂个大概,基本功仍需努力.  
https://github.com/xiaofengShi/CHINESE-OCR
https://github.com/SeanNaren/deepspeech.pytorch  
可能到时候基于这些修改  


ctc也有其它的魔改版本,比如transducer,以及百度的一些魔改,引用[知乎上面大佬的话](https://www.zhihu.com/question/279927514):
>仔细读了 Transducer 的论文，感觉就是 Alex Graves 对 CTC 的魔改，嫌 CTC loss 没有建模输出的各个时间步之间的依赖，于是引入了一个新的 RNN 建模这个，同时类比 CTC 的训练算法搞了一个对齐网格做动态规划来高效训练。  
如此说来的话，基本可以断定 Transducer 优于 CTC 了，除非是为了模型的简单性或者数据量太少导致 Transducer 中的语言模型训不好，否则无脑上 Transducer 就行了。  
至于建模 CTC 各步输出之间的关系，我以前想过搞一个转移矩阵（其实 LSTM + CTC + CRF = =），现在想想应该还是 Transducer 的建模能力更强一些，但是 LSTM-CTC-CRF 的好处是解码不用做 beam search 这种近似搜索，可以求得全局最优解，不知道为什么语音识别里没人用 LSTM-CTC-CRF（或者仅仅是我不知道而已）。

## lstm+attention篇  
谷歌的机器翻译等是用的这玩意,还有很多图片上的模式识别,ocr用的这种模型.但是因为不能进行流式传输,就没有再深入研究.可能以后有兴趣或机会的话再研究吧.


------
## 模型选择与思考
因为要求实时识别,注意到[这篇论文](https://arxiv.org/pdf/1707.07413.pdf)中开头给出的结论:
> The alignment between input and output units is monotonic. This is a reasonable assumption for the ASR
task, which enables models to do streaming transcription. CTC and RNN-Transducers make this assumption, but Attention models do not.  
输入和输出单元之间的对齐是单调的。这是 ASR任务的合理假设，该任务使模型能够进行流式转录。CTC 和 RNN-Transducers 做出了这个假设，但是 Attention 模型却没有。  

关于asr任务的资料:
>在语音识别（Automatic Speech Recognition, ASR）中，常用的评估标准为词错误率WER（Word Error Rate），当测试语言为中文时，也会使用CER（Character Error Rate）字符错误率，两者计算原理是一样的，英文基本单位是单词Word，中文基本单位是Character。本文中统一使用WER。  
WER计算方式为：为了使识别出来的词序列和标准的词序列之间保持一致，需要进行替换，删除，或者插入某些词。这些插入，替换，删除的词的总个数，除以标准的词序列中词的个数的百分比，即为WER，其计算公式如下所示：  
字准确率Word Accuracy，简写为 W.Acc，则有W.Acc计算公式：W.Acc = 1 – WER 
由于存在计算关系，所以我们测试时只需要测WER即可。


------
## 当前各类成熟算法比较

### 百度的deepspeech
DeepSpeech是国内百度推出的语音识别框架，目前已经出来第三版了。不过目前网上公开的代码都还是属于第二版的。

(1) DeepSpeech V1

其中百度研究团队于2014年底发布了第一代深度语音识别系统 Deep Speech 的研究论文，系统采用了端对端的深度学习技术，也就是说，系统不需要人工设计组件对噪声、混响或扬声器波动进行建模，而是直接从语料中进行学习。团队采用 7000 小时的干净语音语料，通过添加人工噪音的方法生成 10 万小时的合成语音语料，并在 SWITCHBOARD评测语料上获得了 16.5% 的 WER（词错误率，是一项语音识别的通用评估标准）。当时的实验显示，百度的语音识别效果比起谷歌、Bing 与 Apple API 而言优势明显。

(2) DeepSpeech V2

 2015 年年底，百度 SVAIL 推出了Deep Speech 2，它基于 LSTM-CTC （Connectionist Temporal Classification）的端对端语音识别技术，通过将机器学习领域的 LSTM 建模与 CTC 训练引入传统的语音识别框架里，提出了具有创新性的汉字语音识别方法。并能够通过深度学习网络识别嘈杂环境下的两种完全不同的语言——英语与普通话，而端到端的学习能够使系统处理各种条件下的语音，包括嘈杂环境、口音及区别不同语种。而在 Deep Speech 2 中，百度应用了 HPC 技术识别缩短了训练时间，使得以往在几个星期才能完成的实验只需要几天就能完成。在基准测试时，系统能够呈现与人类具有竞争力的结果。

(3) DeepSpeech V3

2017年10月31日，百度的硅谷AI实验室发布了Deep Speech 3，这是下一代的语音识别模型，它进一步简化了模型，并且可以在使用预训练过的语言模型时继续进行端到端训练。并开发了Cold Fusion，它可以在训练Seq2Seq模型的时候使用一个预训练的语言模型。百度在论文中表明，带有Cold Fusion的Seq2Seq模型可以更好地运用语言信息，带来了更好的泛化效果和更快的收敛，同时只需用不到10%的标注训练数据就可以完全迁移到一个新领域。Cold Fusion还可以在测试过程中切换不同的语言模型以便为任何内容优化。Cold Fusion能够用在Seq2Seq模型上的同时，它在RNN变换器上应当也能发挥出同样好的效果。

>论文题目: Deep Speech 2: End-to-End Speech Recognition in English and Mandarin  
>论文地址: https://arxiv.org/pdf/1512.02595.pdf  
>tensorflow版本: https://github.com/mozilla/DeepSpeech  
>pytorch版本: http://www.github.com/SeanNaren/deepspeech.pytorch

---
# 时间节点记录
20.1.29 开始研究模板代码,尝试修改.  
1. 从https://github.com/xiaofengShi/CHINESE-OCR 开始下手  

20.1.30 
1. 换目标 之前的包安装太复杂,更为  
https://github.com/Diamondfan/CTC_pytorch  
2. 分析完毕 models/model_ctc.py 
3. 下次开始学习训练数据的读取与加载  
https://blog.csdn.net/qq_27825451/article/details/96130126

20.2.8  
1. 完成dataloader数据加载与读取。  
20.2.9  
1. 完成网络结构搭建
20.2.10
1. 完成网络结合数据启动。准备进行数据预处理与联合调试
ps：辽宁的考研成绩已经出了，有点紧张，数学考崩了
# 学习到的操作
1. editdistance 和python_levenshtein 包之间的区别  
ed.eval()与ls.distance()应该没啥区别....应该可以相互替换,计算的都是编辑距离.  
 
2. python 小技巧  
```
c = 23 if a == 2 else 45
```  

3. bidirectional超参数,可将lstm,rnn设置为双向传递
4. 参数中的偏置什么时候可以设为false  
https://blog.csdn.net/u013289254/article/details/98785869
5. pytorch lstm的参数分析  
https://zhuanlan.zhihu.com/p/39191116
6. pytorch 构造函数  
https://blog.csdn.net/qq_27825451/article/details/90550890  
https://blog.csdn.net/qq_27825451/article/details/90705328  
7. Pytorch之contiguous函数  
https://zhuanlan.zhihu.com/p/64376950  

```
x.unsqueeze(-1) #指定位置增加一维度,长度为1
x.squeeze(-1) #指定位置减小一长度为1的多余维度
```  
8. assert 
```
a=-1
#报错
assert a>0,"a超出范围"
#正常运行
assert a<0
```

# 届不到的操作
1. 为什么要骚一下?  models/model_ctc.py line33
```
x = x.transpose(-1, -2)
x = self.batch_norm(x)
x = x.transpose(-1, -2)
```
---
# 阅读过的论文
Exploring Neural Transducers for End-to-End Speech Recognition:https://arxiv.org/abs/1707.07413