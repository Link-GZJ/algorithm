# algorithm

算法学习

## 2023-7-13

### Lc4 二分算法

寻找两个有序数组的中位数，时间复杂度要求log(m+n)。思路：寻找两个有序数组第k(m+n/2)小的数，需要区分奇偶，分别寻找两个数组k/2 -1位置的元素， 若a[k/2 -1] <= b[k/2-1] 则丢掉a[k/2 - 1]
及它以前的元素，否则，丢掉b[k/2 -1]及以前的元素，k则减去丢掉的元素，下次只需寻找第（k-丢掉元素个数） 位置的元素。边界情况 注意k/2-1
是否大于数组长度，若大于，则取另外数组k位置的元素即可，k==1时，则比较两个数组的第一个元素，谁更小，谁符合要求返回。

### Lc11 双指针

当a[left] <= a[right]时，抛弃left,left++,因为 area = min(a[left],a[right]) * (right-left);如果抛弃right,则后面的值不会大于area,只有移动较小的一方，
才有可能比上次a[min]大。

### lc15 双指针

首先要数组排序，如果a[i] 和 a[i - 1]相等，直接跳过即可。关键条件 排序， a[first] + a[second] + a[third] = 0; 时间复杂度 O(N`2) 空间复杂度 O(N)

### lc16 双指针

解法与15类似

### lc18 26 27 双指针

解法与15类似

### lc31 下一个排列 双指针

左边i寻找一个较小数，右边i+1需找一个较大数，较小数尽量靠右，较大数尽可能小，则i+1 到n为降序，而后从右边再照一个数>a[i],进行交换，i+1后的序列再进行反转，变为升序

### lc33 旋转数组

这里的关键是不论如何总有一边是有序的

### lc34 二分

两次二分查找即可，第一次遍历左边，遍历一遍即为target第一次出现的位置，相反，遍历右边一遍，即为target最后一次的位置

### lc35 二分

left最后指向的一定是第一个大于等于target的位置

### lc36 有效的数独

问题的关键在每个小九宫格，用所在行/3,所在列/3 即可表示

### lc1991 寻找数组的中心索引

比较简单，遍历数组的每个元素，元素左边之和与元素右边之和相等，即满足条件

### lc56 合并区间

重点先排序，排序之后那么可合并的区间一定是连续的，其次是合并规则，想象两条线如果相交，则第一条线的end >= 第二条线的start

### lc48 矩阵旋转

重点要求不能使用额外的内存，只能进行元素交换，首先可以找到一个公式 a[j][n-1-i] = a[i][j]; 即为a[i][j]旋转后的位置，每次旋转90°，每个元素旋转4次可以回到原点，因此每一轮都可以进行四次元素交换，就这样可以
得到4个交换公式。那些元素需要交换？分为奇偶，偶数为i<n/2,j<n/2.奇数为i<n/2,(实际上为n-1/2,这里便于合并)，j<n+1/2;

### lc 零矩阵

需要两个数组分别记录为0的行和列，再遍历更新即可，空间复杂度为i+j,时间复杂度为i*j

### lc498 对角线遍历

需要找规律，根据对角线遍历，区分奇偶，没太理解

### lc14 最长公共前缀

写了两种，横向扫描和纵向扫描

### lc5 最长回文串

### lc141、142、160、19、21、23、86、876关于单链表算法的一些技巧

> 参考自labuladong的算法笔记： https://labuladong.github.io/algo/

#### 1.合并两个有序链表<br>

- 这里会用到一个虚拟头节点dummy,可以简化边界情况的处理

#### 2.单链表的分解

- 这里每次迭代的时候，直接p=p.next是不可行的，分解后的两个链表可能存在环,以下是正确写法

```java
Node temp=p.next;
        p.next=null;
        p=temp;
```

#### 3.合并k个有序链表

- 这里需要使用到一个特殊的数据结构，优先级队列，java中已有实现--**PriorityQueue**,底层数据结构是二叉堆，一种特殊的完全二叉树，由数组实现<br>
- 优先级队列的重要操作:swim(上浮)，sink(下沉)，时间复杂度是log(k),k是优先级队列中的元素个数

#### 4.单链表的倒数第k个节点

- 正常的解法是一共遍历两次链表，第一次得到链表长度，然后遍历第n-k+1个节点<br>
- 如果只让遍历一次呢？<br>
- 先让一个指针走k步，那么剩余n-k可走，这时让另一个指针从头节点出发，与第一个指针一起走，第一个指针走到尽头时，第二个指针走了n-k步，它恰好就是倒数第k个节点.这样一次遍历就解决了问题。

#### 5.单链表的中点

一般想法也是遍历两次。如何遍历一次呢？<br>

- 使用快慢指针同时指向头节点，每当慢指针前进一步，快指针就前进两步，这样当快指针走到终点时，慢指针就到达中点
- 需要注意的是，元素个数为偶数时，慢指针指向的是中间两个的后一个元素

#### 6.判断链表中是否包含环，及得到环的起点

- 使用5中的算法，当快慢指针相遇时，说明存在环
- 快慢指针相遇后，假设slow走了k步，则fast走了2k步，多走的k步=环内的距离，设相遇的地点距离环起点为m,则头节点到环起点的距离为k-m，让slow = head,那么slow到
  环起点的距离为k-m，而fast到环起点的距离也恰好为k-m，当它们再次相遇时，就是环的起点

#### 7.两个链表是否相交

- 解决这个问题的核心是，让p1、p2通过某些方式同时到达相交点c
- 可以让p1走完A之后继续走B,p2走完B之后走A，理论上他们走的总距离时相等的，当他们相遇时，就是相交点

### lc167、26、27、283、344、5、83 数组问题中的双指针技巧

#### 1.快慢指针技巧

- 原地修改数组问题，或是对某些元素原地删除。另一大类就是滑动窗口算法，这个后面总结

#### 2.左右指针的常用算法

- 二分查找

```java
int binarySearch(int[]nums,int target){
        int left=0,right=nums.length-1;
        while(left<=right){
        int mid=(left+right)/2;
        if(nums[mid]==target){
        return mid;
        }else if(nums[mid]>target){
        right=mid-1;
        }else{
        left=mid+1;
        }
        }
        //未找到
        return-1;
        }
```

- 查找一个排序数组的两数之和
- 反转数组
- 回文串的判断
- 寻找最长回文串，解决问题的关键是，**从中间向两端扩散的双指针技巧**

### 二叉树/递归的核心思维

#### 1.二叉树的两种解题思维模式

- 是否可以通过遍历一边二叉树得到答案。**遍历的思维模式**
- 是否可以通过子问题(子树)的答案推导出原问题的答案。**分解问题的思维模式**
- 如果单独抽出一个二叉树节点，可以做什么事情？可以在什么时候做？(前序、中序、后序)

#### 2.深入理解前中后序

- 前中后序是遍历二叉树过程中，处理每一个节点的三个特殊的时间点。二叉树中，每一个节点都有 **唯一** 属于自己的前中后序位置
- 二叉树的所有问题，就是让你在前中后序位置注入巧妙的代码逻辑，去达到自己的目的，你只需要简单的思考当前节点应该做什么，其他的不用管，抛给二叉树遍历框架，递归会在所有节点做相同的操作

#### 3.从树的角度看动归/回溯/DFS算法的区别与练习

他们都属于二叉树的拓展，只是关注点不同<br>

- 动态规划算法属于分解问题的思路，它的关注点在整个 **子树**
- 回溯算法属于遍历的思路，它的关注点在节点间的 **树枝**
- DFS算法属于遍历的思路，它的关注点在 **单个节点**

#### 4.层序遍历

层序遍历属于迭代遍历，代码框架如下：

```java
//输入一个二叉树的根节点，层序遍历这棵二叉树
void levelTraverse(TreeNode root){
        if(root==null)return;
        Queue<TreeNode> q=new LinkedList<>();
        q.offer(root);
        //从上到下遍历二叉树的每一层
        while(!q.isEmpty()){
        int sz=q.size();
        //从左至右遍历同一层的每一个节点
        for(int i=0;i<sz; i++){
        TreeNode cur=q.poll();
        //将下一层节点放入队列
        if(cur.left!=null){
        q.offer(cur.left);
        }
        if(cur.right!=null){
        q.offer(cur.right);
        }
        }
        }
        }
```

### 动态规划的解题框架

- 动态规划问题的一般形式是求最值，核心问题是穷举。状态转移方程、最优子结构、重叠子问题是动态规划三要素。
- 明确base case -> 明确 **[状态]** -> 明确 **[选择]** -> 定义dp数组/函数的含义<br>
  代码框架:

```
# 自顶向下递归的动态规划
def dp(状态1, 状态2, ...):
    for 选择 in 所有可能的选择:
        # 此时的状态已经因为做了选择而改变
        result = 求最值(result, dp(状态1, 状态2, ...))
    return result

# 自底向上迭代的动态规划
# 初始化 base case
dp[0][0][...] = base case
# 进行状态转移
for 状态1 in 状态1的所有取值：
    for 状态2 in 状态2的所有取值：
        for ...
            dp[状态1][状态2][...] = 求最值(选择1，选择2...)

```

- 递归算法的时间复杂度：子问题的个数 * 解决一个子问题需要的时间
- 动态规划问题的解法：1.暴力递归(一般会超出时间限制) 2.带备忘录的递归解法(自顶向下) 3.dp数组的迭代解法(自底向上)

### 回溯算法的解题框架

- 抽象的说解决一个回溯问题，实际上就是在遍历一棵决策树，树的每个叶子节点放着一个合法答案。
- 在回溯树的一个节点上思考三个问题
    - 路径：已经做出的选择
    - 选择列表：当前可以做的选择
    - 结束条件： 到达决策树的底层，无法再做选择
- 核心就是递归之前做选择，递归之后撤销选择,代码框架

```
def backtrack(...):
    for 选择 in 选择列表:
        做选择
        backtrack(...)
        撤销选择
```

- 抽象出来的回溯树，也可以叫决策树，每个节点都在做决策，上文也说过回溯算法的重点在 **树枝**

> ![img.png](resource/backtrack.png)

### 回溯算法 -- 排列、组合(子集)问题

- 形式一：元素无重复不可复选

```java
//组合(子集)问题回溯算法框架
void backtrack(int[]nums,int start){
        for(int i=start;i<nums.length;i++){
        //做选择
        track.add(num[i]);
        //注意参数
        backtrack(nums,i+1);
        //撤销选择
        track.removeLast();
        }
}

//排列问题回溯算法框架
void backtrack(int[]nums){
        for(int i=0;i<nums.length;i++){
        //剪枝逻辑
        if(used[i]){
        continue;
        }
        //做选择
        track.add(num[i]);
        used[i]=true;
        //注意参数
        backtrack(nums);
        //撤销选择
        track.removeLast();
        used[i]=false;
        }
}
```

- 形式二：元素无重复可复选

```java
//组合(子集)问题回溯算法框架
void backtrack(int[]nums,int start){
        for(int i=start;i<nums.length;i++){
        //做选择
        track.add(num[i]);
        //注意参数
        backtrack(nums,i);
        //撤销选择
        track.removeLast();
        }
}

//排列问题回溯算法框架
void backtrack(int[]nums){
        for(int i=0;i<nums.length;i++){
        //做选择
        track.add(num[i]);
        //注意参数
        backtrack(nums);
        //撤销选择
        track.removeLast();
        }
}
```

- 形式三：元素可重复不可复选

```java
//组合(子集)问题回溯算法框架
void backtrack(int[]nums,int start){
        for(int i=start;i<nums.length;i++){
        //剪枝逻辑，跳过值相同的相邻树枝
        if(i>start&&nums[i]==nums[i-1]){
        continue;
        }
        //做选择
        track.add(num[i]);
        //注意参数
        backtrack(nums,i+1);
        //撤销选择
        track.removeLast();
        }
}

//排列问题回溯算法框架
void backtrack(int[]nums){
        for(int i=0;i<nums.length;i++){
        //剪枝逻辑
        if(used[i]){
        continue;
        }
        //剪枝逻辑，固定相同的元素在排列中的相对位置
        if(i>0&&nums[i]==nums[i-1]&&!used[i-1]){
        continue;
        }
        //做选择
        track.add(num[i]);
        used[i]=true;
        //注意参数
        backtrack(nums);
        //撤销选择
        track.removeLast();
        used[i]=false;
        }
}
```

- 组合(子集)问题，重点是迭代开始用start,就是通过保证元素的相对顺序，防止出现重复的子集

```java
//元素不可复选
for(int i=start;i<length; i++){
        backtrack(i+1,...)
        }
//元素可复选
        for(int i=start;...){
        backtrack(i,...)
        }
//元素重复 首先要排序，使重复元素相邻
        Arrays.sort();
        for(...){
        //裁剪相邻相等的树枝
        if(i>start&&nums[i]==nums[i-1]){
        continue;
        }
}
```

- 排列问题，通过boolean[] 数组控制访问过的元素,如果元素可复选，则去掉boolean[]数组即可

```java
boolean[]used=new boolean[nums.length];
        for(int i=0;i<nums.length;i++){
        if(used[i]){
        continue;
        }
        //如果元素重复(同样需要先排序)，则通过固定相等元素的位置，来保证不会走重复的树枝，！used[i-1]保证i-1始终在前面
        if(i>0&&nums[i]==nums[i-1]&&!used[i-1]){
        continue;
        }
}
//对于元素重复，还有一种更易理解的方法
// 初始化一个数组中不存在的元素
int preNum=-999;
for(...){
        //原理使裁剪掉同一层相邻且相等的树枝
        if(nums[i]==preNum){
        continue;
        }
        preNum=nums[i];
}
```

### BFS算法框架
- 常见场景：在一幅“图”中找到从起点start不断扩散到终点target的最近距离(和树的层序遍历很像)，重要的数据结构：队列，代码框架：

```java
//计算起点start到终点target的最近距离
int BFS(Node start,Node target){
    Queue<Node> q;//核心数据结构
    Set<String> visited; //避免走回头路
    q.offer(start);
    visited.add(start);
    while(!q.isEmpty){
        //临时变量，因为q的size会变
        int sz = q.size();
        //将当前队列的节点向四周扩散
        for(int i = 0; i < sz; i++){
            Node cur = q.poll();
            if(cur == target){
                return step;    
            }
            //将cur的相邻节点加入队列
            for(Node x : cur.adj()){
                if(!visited.contains(x)){
                    q.offer(x);
                    visited.add(x);
                }       
            }
        }
        step++;
    }
    //走到这里，说明无解，不能走到目标节点
}
```
- 知道就行：双向BFS优化，与传统的BFS时间复杂度相同，但少遍历一些节点，效率确实快一些，使用限制的必须知道终点target

>![img.png](resource/bfs1.png)
> ![img_1.png](resource/bfs2.png)

```java
//计算起点start到终点target的最近距离
int BFS(Node start,Node target){
    Set<String> q1;
    Set<String> q2;
    Set<String> visited; //避免走回头路
    q1.offer(start);
    q2.offer(target);
    while(!q1.isEmpty && !q2.isEmpty()){
        //哈希集合在遍历的过程中不能修改，用 temp 存储扩散结果
        Set<Node> temp;
        //将当前队列的节点向四周扩散
        for(String cur : q1){
            if(q2.contains(cur)){
                return step;    
            }
            //这个位置滞后，否则q1 和 q2永远不会相遇
            visited.add(x);
            //将cur的相邻节点加入
            for(Node x : cur.adj()){
                if(!visited.contains(x)){
                    temp.add(x);
                }       
            }
        }
        step++;
        //这里交换 q1 q2，下一轮 while 就是扩散 q2
        q1 = q2;
        q2 = temp;
    }
    //走到这里，说明无解，不能走到目标节点
}
```