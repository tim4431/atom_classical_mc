我想 model在neutral atom computer中，使用moving tweezer (例如aod) 把atom从slm trap中拽出来对应的 atom heating 和 loss，我希望计算对应某一个aod intensity和position 的ramping sequence，计算atom loss / heating

大体的计算思路如下：define一个3d space，给定tweezer参数生成trap potential；以某个初始温度的分布，来模拟 trap 在运动的过程中空间势能的变化，将atom（以Rubidium87 atom为例）的运动使用classical monte carlo 来模拟

请使用英文给出plan，放置在
plan.md
 ，然后使用英文编写python （或者混合c）代码用于模拟的核心代码，放在/src文件夹