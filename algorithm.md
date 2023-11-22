# 算法总结

## 二分查找法

使用该算法的前提是序列有序，且最终的查找结果只能是一个，时间复杂度一般为logn；边界问题left<=right,暂且死记

## 回溯算法

## 动态规划

核心：记住已解决的子问题的解 有两种方式保存求解方式： 1.自顶向下的备忘录法 需要一个数组来保存每一次计算过的子问题的解

```java
public int Fibonacci(int n)
        {
        if(n<=0)
        return n;
        int[]Memo=new int[n+1];
        for(int i=0;i<=n;i++)
        Memo[i]=-1;
        return fib(n,Memo);
        }
public int fib(int n,int[]Memo)
        {
        if(Memo[n]!=-1)
        return Memo[n];
        //如果已经求出了fib（n）的值直接返回，否则将求出的值保存在Memo备忘录中。				
        if(n<=2)
        Memo[n]=1;
        else Memo[n]=fib(n-1,Memo)+fib(n-2,Memo);
        return Memo[n];
        }
```

2.自底向上 先计算子问题，再计算父问题，利用循环计算n次，可大大减少利用空间 参考博客：https://blog.csdn.net/u013309870/article/details/75193592

```java
public int fib(int n)
        {
        if(n<=0)
        return n;
        int[]Memo=new int[n+1];
        Memo[0]=0;
        Memo[1]=1;
        for(int i=2;i<=n;i++)
        {
        Memo[i]=Memo[i-1]+Memo[i-2];
        }
        return Memo[n];
        }
//再简化        
public int fib(int n)
        {
        if(n<=1)
        return n;

        int Memo_i_2=0;
        int Memo_i_1=1;
        int Memo_i=1;
        for(int i=2;i<=n;i++)
        {
        Memo_i=Memo_i_2+Memo_i_1;
        Memo_i_2=Memo_i_1;
        Memo_i_1=Memo_i;
        }
        return Memo_i;
        }
```