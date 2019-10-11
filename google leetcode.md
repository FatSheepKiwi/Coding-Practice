
## LeetCode题
大佬整理的题单
https://leetcode.com/list/cn1cvit/
涵盖了1-36题
another 大佬的题单，重复的标记标记
https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=446944&ctid=201329

1. 21.Merge Two Sorted Lists
[LeetCode](https://leetcode.com/problems/merge-two-sorted-lists/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/78484941)

出现面经：
[狗家技术电面-十二月第一波](https://www.1point3acres.com/bbs/thread-462577-1-1.html)
第一题
给你两个sorted LinkedList的head，要求merge两个linked list 然后依然是non-descending order.
follow-up sort k lists， 55题

key word：Linked list, Recursion

thought：
将两个已经排序了的链表合并，返回新的头。递归解决，如果一条没了，返回另一条；否则根据大小，更新next后返回。
It is about merger two sorted LinkedList. We could use recursion to solve this problem. If either of the list is null, return another one. Then compare their value, return smaller one, and add its next node to function to get the rest list.

code：
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null){
            return l2;
        }
        if (l2 == null){
            return l1;
        }
        if (l1.val < l2.val){
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        }
        else{
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }
}
```

2. 62.Unique Paths (高频 9)
[LeetCode](https://leetcode.com/problems/unique-paths/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/80161526)

key word：Array, DP

出现面经：
[google 面试频率最高的题目大全（频率统计）](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=446944&ctid=201329)
著名变种：DP。给定一个矩形的长宽，用多少种方法可以从左上角走到右上角 （每一步，只能向正右、右上 或 右下走）：整个矩形遍历做DP即可，不需要想复杂. 牛人云集,一亩三分地
-follow up：如果给矩形里的三个点，要求解决上述问题的同时，遍历这三个点 （切割矩形，一个一个地做DP，然后相加）
-follow up：如何判断这三个点一个是合理的，即存在遍历这三个点的路经
-follow up：如果给你一个H，要求你的路径必须向下越过H这个界，怎么做 （别问我，我不会）
楼主这题试一试用镜像做，也就是说重点不在(W, 0), 而在(W, 2H)， W是矩阵的宽度

thought：
从巨岑左上角到右下角的不同的路径的个数。
A simple DP problem. Using a two dimensional dp array to solve it. Set initial value 1 to dp[1][1] then start iteration. Using a bigger dp array is for avoid our of bound.

code：
O(m * n) time & space solution
```java
class Solution {
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m + 1][n + 1];
        dp[1][1] = 1;
        for (int i = 1; i <= m; i++){
            for (int j = 1; j <= n; j++){
                dp[i][j] += dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m][n];     
    }
}
```
O(n) space solution.
```java
class Solution {
    public int uniquePaths(int m, int n) {
        int[] dp = new int[n + 1];
        dp[1] = 1;
        for (int i = 1; i <= m; i++){
            for (int j = 1; j <= n; j++){
                dp[j] += dp[j - 1];
            }
        }
        return dp[n];     
    }
}
```

3. 68.Text Justification (高频 3)
[LeetCode](https://leetcode.com/problems/text-justification/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/83905310)

key word：String

出现面经：
[Google技术电面2019 Intern总结 求大米打赏](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=468194&extra=page%3D1%26filter%3Dsortid%26sortid%3D311%26searchoption%5B3086%5D%5Bvalue%5D%3D9%26searchoption%5B3086%5D%5Btype%5D%3Dradio%26searchoption%5B3088%5D%5Bvalue%5D%3D1%26searchoption%5B3088%5D%5Btype%5D%3Dradio%26searchoption%5B3046%5D%5Bvalue%5D%3D1%26searchoption%5B3046%5D%5Btype%5D%3Dradio%26searchoption%5B3109%5D%5Bvalue%5D%3D1%26searchoption%5B3109%5D%5Btype%5D%3Dradio%26sortid%3D311%26orderby%3Ddateline)
Text Justification的一道题

总之也是磕磕碰碰，没有清晰的思路。


thought：
给一个maxWidth，将数组中的word填进去，然后把符合格式的每一行都是maxWidth的String list 返回。

首先确定这一行能放哪几个单词，index表示此行开始的word，count为此行正常的只有一个空格的长度，last为下一行第一个word。在last不越界的情况下，加上word[last] + count（目前长度） + 1（空格）大于maxwidth了，break，否则count连接上 last 和 一个空格， last++。

然后开始构建这一行，用一个StringBuilder，先加上index，然后算出有几个空 diff （不一定一个空格哦）
如果last已经时最后一个单词了，或者这一行只能放一个单词， diff == 0，那么就正常的 空格 + 单词， 完成这一行
如果不是，那么需要填充，不止一个空格，而且左边要比右边多，space是大方向， remain时剩下的要加上的。然后开始构建，先加上space个空格，然后根据remain还有没加一个，最后加一个空格加单词。

完成之后res加上sb，index = last。

code：
```java
class Solution {
    public List<String> fullJustify(String[] words, int maxWidth) {
        List<String> res = new ArrayList<>();
        int index = 0;
        while (index < words.length){
            //count: count this length of this line's words
            int count = words[index].length();
            //last: index of last word next line
            int last = index + 1;
            
            while (last < words.length){
                //out of bound
                if (words[last].length() + count + 1 > maxWidth){
                    break;
                }
                count += 1 + words[last].length();
                last++;
            }
            
            StringBuilder sb = new StringBuilder();
            //append the frist word of this row
            sb.append(words[index]);
            //get the number of spaces of this row
            int diff = last - 1 - index;
            //if it is the last row, add word with normal space and fill rest space
            //if this line could only contain one word diff == 0.
            if(last == words.length || diff == 0){
                for (int i = index + 1; i < last; i++){
                    sb.append(" ");
                    sb.append(words[i]);
                }
                for (int i = sb.length(); i < maxWidth; i++){
                    sb.append(" ");
                }
            }
            else{//not the last row, we should assign left more than right
                //count every space length
                int spaces = (maxWidth - count) / diff;
                int remain = (maxWidth - count) % diff;
                for (int i = index + 1; i < last; i++){
                    for (int j = spaces; j > 0; j--){
                        sb.append(" ");
                    }
                    if (remain > 0){
                        sb.append(" ");
                        remain--;
                    }
                    sb.append(" ");//a normal space, we should count it inside, because we add it in the count before.
                    sb.append(words[i]);
                }
            }
            res.add(sb.toString());
            index = last;
        }
        return res;
    }
}
```

4. 96.Unique Binary Search Trees (高频 3)
[LeetCode](https://leetcode.com/problems/unique-binary-search-trees/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/79969029)

key word: DP, Tree

出现面经：
如果有n个node，可以构成多少个不同的BST。
DP解决，
BST左右子树也是BST，所以 如果我选择 i 个元素作为root，那么有 dp[i - 1] * dp[n - i - 1]种不同的方式构建。所以这样我就可以用DP，两个for循环来计算选第 i 个元素时第结果了。


thought：
For a binary search tree, its left subtree and right substree are also binary search tree. So, for number from 1 to n, If I choose first element as root node, ways of constructing a binary search tree is dp[0] * dp[n - 1]. For the ith element, it is dp[i - 1] * dp[n - i - 1]. dp[0] = dp[1] = 1. So we could calculate dp[n] using O(n) space, O(n^2) time complexity.

code：
```java
class Solution {
    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        for (int i = 1; i <= n; i++){
            int temp = 0;
            for (int j = 0; j < i; j++){
                temp += dp[j] * dp[i - j - 1];
            }
            dp[i] = temp;
        }
        return dp[n];
    }
}
```

5. 253.Meeting Rooms II (高频 4)
[LeetCode](https://leetcode.com/problems/meeting-rooms-ii/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/82980492)
[252. Meeting Rooms](https://leetcode.com/problems/meeting-rooms/description/)

key word:Heap, Greedy, Sort

出现面经：
[狗家上门](https://www.1point3acres.com/bbs/thread-461424-1-1.html)


thought：
给一个Interval的数组，找到能够同时举办这么多会议的最小会议室数量。
用一个map，如果start了，put 一个+1；end了，put 一个-1；
然后就遍历map中的内容，此时需要的room加上这个时间点需要或者不需要的room的数量，记录最大值。
Using a map to store the start time and end time, its value is the change of rooms needed. Use another iteration to get the max rooms needed.

code：
```java
/**
 * Definition for an interval.
 * public class Interval {
 *     int start;
 *     int end;
 *     Interval() { start = 0; end = 0; }
 *     Interval(int s, int e) { start = s; end = e; }
 * }
 */
class Solution {
    public int minMeetingRooms(Interval[] intervals) {
        Map<Integer, Integer> map = new TreeMap<>();
        for (Interval in : intervals){
            map.put(in.start, map.getOrDefault(in.start, 0) + 1);
            map.put(in.end, map.getOrDefault(in.end, 0) - 1);
        }
        int res = 0, rooms = 0;
        for (Integer time : map.keySet()){
            rooms += map.get(time);
            res = Math.max(res, rooms);
        }
        return res;
    }
}
```
Another Approach:
分别记录开始和结束时间到两个数组，分别排序为升序
初始化count = 1， i是start数组指针，从1开始；j是end指针，从0开始，如果j在i之前，而且end[j] < start[i]，可以正常开会，j move on；如果start[i] < end[j]，得加钱。
Another approach. Sort the start time and end time. Then use a pointer j to denotes the last ended meeting's end time.

Then start from the second element in start, if start after or equal end[j], it means this meeting could have in current count rooms, and j should < I for you can't start a meeting it does not start yet. If start[I] < end[j], it means we need more rooms.

code:
```java
/**
 * Definition for an interval.
 * public class Interval {
 *     int start;
 *     int end;
 *     Interval() { start = 0; end = 0; }
 *     Interval(int s, int e) { start = s; end = e; }
 * }
 */
class Solution {
    public int minMeetingRooms(Interval[] intervals) {
        if (intervals == null || intervals.length == 0){
            return 0;
        }
        int n = intervals.length;
        int[] start = new int[n];
        int[] end = new int[n];
        for (int i = 0; i < n; i++){
            start[i] = intervals[i].start;
            end[i] = intervals[i].end;
        }
        Arrays.sort(start);
        Arrays.sort(end);
        int count = 1;
        for (int i = 1, j = 0; i < n; i++){
            if (j < i && start[i] >= end[j]){
                j++;
            }
            else if( start[i] < end[j]){
                count++;
            }
        }
        return count;
    }
}
```

6. 312.Burst Balloons (高频 3)
[LeetCode](https://leetcode.com/problems/burst-balloons/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/83504745)
[discussion](https://leetcode.com/problems/burst-balloons/discuss/76228/Share-some-analysis-and-explanations)

key word: D&C, DP

出现面经：

thought：
n个气球，爆破一个可以得到nums[left] * nums[i] * nums[right]钱，最多能得多少钱。

首先给nums左右两边各加上一个边界1，形成newNums

然后开始dp，dp[i][j] 表示如果以i 和 j 为左右边界（也击破它们），我们能得到的最多的钱。
第一个for循环，确定window size，也就是我们这次处理的气球数，从2开始，也就是3个气球；最多到n - 1，也就是n个
第二个for循环，确定开始击破气球的位置，left = 0，最多到n - k；
第三个for循环，计算right右边界，开始dp，
dp[left][right] = max(itself, i作为最后一个击破的元素 + dp[left][i] + dp[i][right]);

最后返回dp[0][n - 1]；

Add fake ballons with coin = 1 to head and tail. Then we keep a window start from size 2 and perform dynamic programming. Let i be the last ballon we burst, and left and right are its adjacent ballons.
dp[left][right] = max(dp[left][right], newNums[left] * newNums[i] * newNums[right] + dp[left][i] + dp[i][right]));

code：
```java
class Solution {
    public int maxCoins(int[] nums) {
        int[] newNums = new int[nums.length + 2];
        int n = 1;
        for (int i : nums){
            newNums[n++] = i;
        }
        newNums[0] = newNums[n++] = 1;
        
        int[][] dp = new int[n][n];
        for (int k = 2; k < n; k++){
            for (int left = 0; left < n - k; left++){
                int right = left + k;
                for (int i = left + 1; i < right; i++){
                    dp[left][right] = Math.max(dp[left][right], 
                    newNums[left] * newNums[i] * newNums[right] + dp[left][i] + dp[i][right]);
                }
            }
        }
        return dp[0][n - 1];
    }
}
```

7. 334.Increasing Triplet Subsequence (高频 3)
[LeetCode](https://leetcode.com/problems/increasing-triplet-subsequence/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/84138505)

key word:

出现面经：
问未排序的数组中是否存在连续三个数，它们是递增的

那么找两个指针，n1指向第一个数，先判断， n2指向第二个，如果这俩都不再更新，碰到更大的了，返回

thought：
We want to find out whether there is a inreasing triplet in the array. We just need two pointers to store the two smaller number. Then in the iteration, if there is a number bigger than the two number, the triplet exists.

code：
```java
class Solution {
    public boolean increasingTriplet(int[] nums) {
        int n1 = Integer.MAX_VALUE, n2 = Integer.MAX_VALUE;
        for (int n : nums){
            if (n <= n1){
                n1 = n;
            }
            else if (n <= n2){
                n2 = n;
            }
            else{
                return true;
            }
        }
        return false;
    }
}
```

8. 337.House Robber III (高频 3)
[LeetCode](https://leetcode.com/problems/house-robber-iii/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/84207893)

key word: Tree, DFS

出现面经：

thought：
二叉树，只能隔一个偷一个，如何使偷盗的值和最大。

贪心思想，我们如果偷root，那就在left和right存在情况下偷它们的孩子们（4个），然后和不偷root，在左右递归的值比较，返回更大的。
**Approach 1**: Greedy method. Since we want to get the most money start from root, we also want to do it at left subtree and right subtree. Thus, we could solve this problem recursively.

So the termination codition could be root == null, since this path is end. For the recurrence relation, it is depend on root. If rob root, we have to rod grandchild. If not, just child. So we could get the recursion function.

But this solution is quite slow. It runs 771ms.

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public int rob(TreeNode root) {
        if (root == null){
            return 0;
        }
        int val = root.val;
        if (root.left != null){
            val += rob(root.left.left) + rob(root.left.right);
        }
        if (root.right != null){
            val += rob(root.right.left) + rob(root.right.right);
        }
        return Math.max(val, rob(root.left) + rob(root.right));
    }
}
```
Time Complexity: O(2^n)
Space Complexity: O(n)


DP思想，如果我么偷了这个点，就存进map中，方便以后查询；依然是偷这个node或者不偷，比较出最大的。
**Approach 2**: Dynamic programming. Because we have to calculate overlaping value, we could use dynamic programming to get easier approach. Use a hash map to record the results for visited subtrees.

#### Code
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public int rob(TreeNode root) {
        return robHelper(root, new HashMap<TreeNode, Integer>());
    }
    
    private int robHelper(TreeNode node, Map<TreeNode, Integer> map){
        if (node == null){
            return 0;
        }
        if (map.containsKey(node)){
            return map.get(node);
        }
        int val = 0;
        if (node.left != null){
            val += robHelper(node.left.left, map) + robHelper(node.left.right, map);
        }
        if (node.right != null){
            val += robHelper(node.right.left, map) + robHelper(node.right.right, map);
        }
        
        val = Math.max(val + node.val, robHelper(node.left, map) + robHelper(node.right, map));
        map.put(node, val);
        return val;
    }
}
```

Time Complexity: O(n)
Space Complexity: O(n)

因为只有两种情况，所以用一个数组存就行。
**Approach 3**: We just have two scenarios, so we could simply using a array with two element to store the output of each node.
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public int rob(TreeNode root) {
        int[] res = robHelper(root);
        return Math.max(res[0], res[1]);
    }
    
    private int[] robHelper(TreeNode node){
        if (node == null){
            return new int[2];
        }

        int[] left = robHelper(node.left);
        int[] right = robHelper(node.right);
        int[] res = new int[2];
        
        res[0] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
        res[1] = node.val + left[0] + right[0];
        return res;
    }
}
```


9. 340.Longest Substring with At Most K Distinct Characters (高频 3)
[LeetCode](https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/84208770)

key word: Sliding Window

出现面经：
[骨骼背靠背店面跪经](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=461379&extra=page%3D2)

[ 狗家新鲜店面](https://www.1point3acres.com/bbs/thread-443586-1-1.html)
刚刚结束，咩有OA直接电面
找出string里 max substring with two distinct char

"aaabbb" -> "aaabbb"
"aaaa" -> False
"abbccd" -> "bbcc"

用两个pointer扫一下就好，但是我讲的不太好。。

[狗家面经](https://www.1point3acres.com/bbs/thread-429694-1-1.html?_dsign=518c1cf9)
第三面是换的一个面试官，感觉不经常面试，出了个利口散。 感觉还没有我熟，解释了好久。浪费挺多时间。


thought：
给一个String，找到只由k个不同字符组成的最长substring的长度。

code：
滑动窗口的思想，用快指针j表示现在窗口的结束，i表示窗口开始，map存储每个字符出现的频率
然后开始遍历String，统计出现的频率，如果字符第一次出现，count++
如果count > k了，意味着我们要将i向前移动，去掉一种字符。先讲map[i]--，然后看减到0了么，到了就count--结束循环；没到就继续i++，move on。
每次j的循环都计算一下最大值，结束循环返回。
**Approach 1**: Sliding window. Maintain a sliding window with size k to get the max length. Using a int array to store the frequency of each character in s. Then j is the fast pointer in the right, I is the slow pointer in the left, count is the number of characters current used. If map do not contain this word, store its frequence and count++; If we have more than k words, 

```java
class Solution {
    public int lengthOfLongestSubstringKDistinct(String s, int k) {
        int[] map = new int[256];
        int i = 0, count = 0, res = 0;
        for (int j = 0; j < s.length(); j++){
            if (map[s.charAt(j)]++ == 0){
                count++;
            }
            while (count > k){
                if (--map[s.charAt(i++)] == 0){
                    count--;
                }
            }
            res = Math.max(res, j - i + 1);
        }
        return res;
    }
}
```

more readable solution 
```java
public int lengthOfLongestSubstringKDistinct(String s, int k) {
        int[] bits = new int[256];
        int max = 0, index = 0;
        int count = 0;
        for(int i = 0; i < s.length(); i++) {
            if(bits[s.charAt(i)] == 0) {
                count++;
            }
            bits[s.charAt(i)]++;
            while(count > k) {    // count > k delete char from the substring
                bits[s.charAt(index)]--;
                if(bits[s.charAt(index)] == 0) {
                    count--;
                }
                index++;
            }
            max = Math.max(max, i - index + 1);
        }
        return max;
    }
```
Time Complexity: O(n)
Space Complexity: O(256)


**Approach 2**: Using tree map to store the last occurance. But it is slower.

```java
class Solution {
    public int lengthOfLongestSubstringKDistinct(String s, int k) {
        if (s == null || s.isEmpty() || k == 0){
            return 0;
        }
        //lastOccurance stores the last show of index i's character
        TreeMap<Integer, Character> lastOccurance = new TreeMap<>();
        //inWindow stores the char and the index of current window.
        Map<Character, Integer> inWindow = new HashMap<>();
        int i = 0; 
        int res = 1;
        for (int j = 0; j < s.length(); j++){
            char in = s.charAt(j);
            //get new input char and if exceed window size, remove it.
            while (inWindow.size() == k && !inWindow.containsKey(in)){
                int first = lastOccurance.firstKey();
                char out = lastOccurance.get(first);
                inWindow.remove(out);
                lastOccurance.remove(first);
                i = first + 1;
            }
            //update or add new char in two maps.
            if (inWindow.containsKey(in)){
                lastOccurance.remove(inWindow.get(in));
            }
            inWindow.put(in, j);
            lastOccurance.put(j, in);
            res = Math.max(res, j - i + 1);
        }
        return res;
    }
}
```

10. 394.Decode String (高频 3)
[LeetCode](https://leetcode.com/problems/decode-string/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/83059255)

key word: Stack, DFS

出现面经：
[狗家11月7号on-site 已签](https://www.1point3acres.com/bbs/thread-462598-1-1.html)
第四轮 
国人小哥 
decode string lc上有原题 
input string 2[ab3[cbc]]
output string abcbccbccbcabcbccbccbc
括号前面是需要重复的次数 括号里面是需要重复的string 
stack 秒掉 

thought：
频率 + 字符的压缩的字符，解码出来

用两个栈来完成，一个是存次数的，另一个是存外层的String的
四种情况
如果读到数，一个while循环读出这个数，存进栈中
如果读到‘['，我们要进新的一层了，把res入栈然后清为“”，idx++；
如果读到’]‘，这一层结束了，用一个Stringbuilder，读出之前在栈中的外层String，再把本res重复num次跟在后面，id++
其他，就是正常字符，res += char，idx++；

Using two stacks to contain previous string result and multiple number. There are four circumstance. If it is a number, using a while loop to get its value, and store it to countStack; If it is a character, add it to current res String; If it is a '[', we add current res Stirng to resStack and resset it to "" empty string; If it is a ']', we should add current res to the res stored in resStack to form the decoded String.

code：
```java
class Solution {
    public String decodeString(String s) {
        String res = "";
        Stack<Integer> countStack = new Stack<>();
        Stack<String> resStack = new Stack<>();
        int idx = 0;
        while (idx < s.length()) {
            if (Character.isDigit(s.charAt(idx))) {
                int count = 0;
                while (Character.isDigit(s.charAt(idx))) {
                    count = 10 * count + (s.charAt(idx) - '0');
                    idx++;
                }
                countStack.push(count);
            }
            else if (s.charAt(idx) == '[') {
                resStack.push(res);
                res = "";
                idx++;
            }
            else if (s.charAt(idx) == ']') {
                StringBuilder temp = new StringBuilder (resStack.pop());
                int repeatTimes = countStack.pop();
                for (int i = 0; i < repeatTimes; i++) {
                    temp.append(res);
                }
                res = temp.toString();
                idx++;
            }
            else {
                res += s.charAt(idx++);
            }
        }
        return res;
    }
}
```


11. 399.Evaluate Division（高频 21）
[LeetCode](https://leetcode.com/problems/evaluate-division/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/83282560)

key word: Graph

出现面经：
[狗家onsite](https://www.1point3acres.com/bbs/thread-460984-1-1.html)
给一堆货币转换作为输入，以及要做的query。leetcode上有相似度为99%的题

[Google 2019 暑期实习二面+timeline](https://www.1point3acres.com/bbs/thread-462564-1-1.html)
3. 国人小姐姐
转汇率那道，说可以用dfs和union find做，她表示随便选哪种，选了union find，在纸上写就改得乱七八糟了。

[狗家10/31Sunnyvale面经](https://www.1point3acres.com/bbs/thread-458802-1-1.html)
变形
4. 给你一串input，比如：
A -> B
B -> C
X -> Y
Z -> X
。
。
。
然后让你设计一个data structure来存这些关系，最后读完了以后呢要输出这些关系链：[A -> B -> C, Z -> X -> Y]
如果遇到invalid的case就直接raise error，比如你已经有了A->B->C，这时候给你一个D->C或者B->D就是invalid的。
followup我感觉就是他看你什么case没有cover就提出来然后问你怎么改你的代码和data structure
比如遇到A->B两次，再比如遇到环
这题相当开放，面试官说他遇到过4，5种不同的解法，总之就是最好保证insert是O(1), reconstruct是O(n)

[狗家昂赛面经](https://www.1point3acres.com/bbs/thread-462239-1-1.html?_dsign=dedca4c5)
1. 给了若干个国家之间的汇率，比如USD GBP 0.67, GBP YEN 167, YEN EUR 0.007, 输入任意两个国家，求他们之间的汇率，比如USD YEN, 输出就是0.67 * 167, 如果没有match的话输出N/A，我用了BFS，要注意就是图是双向的，因为给定USD GBP 0.67，其实你同时也知道GBP USD的汇率，楼主一开始只见了单向图，最后面试官提醒了一下。


thought：
先给一堆除式，然后给一堆query，问这些数存在么，实际上是构建一张除法关系图，然后判断两个数是否有一条路径在里面

用两个map来构建这张图，String到List<String>是variable到其他variable的联通关系，String到List<Double>是对应的边的权重，也就是除法的结果，先构建图，双向的 ，然后对每一个query进行DFS。
We use two hash maps to contruct a graph according to equations and values. Then variables in equations are nodes in a graph, values are the weight from node a to node b. Pairs map stores the edges of a node, and valuesPair stores the weight information. First using a for loop to iteratively travel equations and values. Adding edges and weights to pairs and values pair.

Then iteratively access queries, using dfs to find a path from query[0] to query[1]. If does not have this path, return 0.0, and then update -1.0 to res.

For the dfs function, we use a hash set to store nodes we have visited. If a circle is formatted or pairs does not contain start, return 0.0. If found end, return current value; For dfs, we get the edge info and weight info of start node, and iteratively travels the edges using dfs and updated node and values. Remember to remove start in set for upper level backtracking.


code：
```java
class Solution {
    private HashMap<String, ArrayList<String>> pairs;
    private HashMap<String, ArrayList<Double>> valuesPair;
    public double[] calcEquation(String[][] equations, double[] values, String[][] queries) {
        pairs = new HashMap<>();
        valuesPair = new HashMap<>();
        for (int i = 0; i < equations.length; i++){
            String[] equation = equations[i];
            //首先看两个点存在么，不存在就加新的点进入两个map
            if (!pairs.containsKey(equation[0])){
                pairs.put(equation[0], new ArrayList<String>());
                valuesPair.put(equation[0], new ArrayList<Double>());
            }
            if (!pairs.containsKey(equation[1])){
                pairs.put(equation[1], new ArrayList<String>());
                valuesPair.put(equation[1], new ArrayList<Double>());
            }
            //加完之后开始加这个equation的关系，双向的，都要加。
            pairs.get(equation[0]).add(equation[1]);
            pairs.get(equation[1]).add(equation[0]);
            valuesPair.get(equation[0]).add(values[i]);
            valuesPair.get(equation[1]).add(1 / values[i]);
        }
        
        double[] res = new double[queries.length];
        //对于每一个query，进行dfs
        for (int i = 0; i < queries.length; i++){
            String[] query = queries[i];
            res[i] = dfs(query[0], query[1], new HashSet<String>(), 1.0);
            //如果没发进行，比如出现了其他没在equation中的点，返回0.0，也就是-1.0
            if (res[i] == 0.0){
                res[i] = -1.0;
            }
        }
        return res;
    }
    
    private double dfs(String start, String end, HashSet<String> set, double value){
        //形成环了，我们之前visit过了||这个点不是在equation中的点，报错
        if (set.contains(start) || !pairs.containsKey(start)){
            return 0.0;
        }
        //找到了，return value
        if (start.equals(end)){
            return value;
        }
        set.add(start);
        
        //get start的邻居，DFS
        ArrayList<String> strList = pairs.get(start);
        ArrayList<Double> valueList = valuesPair.get(start);
        double temp = 0.0;
        for (int i = 0; i < strList.size(); i++){
            temp = dfs(strList.get(i), end, set, value * valueList.get(i));
            //如果找的到就break
            if (temp != 0.0) {
                break;
            }
        }
        //remove 来 回退 很重要。
        set.remove(start);
        return temp;
    }
}
```


12. 418.Sentence Screen Fitting (高频 3)
[LeetCode](https://leetcode.com/problems/sentence-screen-fitting/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/84230161)

key word: DP

出现面经：
[狗狗EP大二面经](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=456367&extra=page%3D2)
利口418！Sentence Screen Fitting。我之前没有做过但是这几天在leetcode上找到了它。我估计前五分钟跟他聊思路，然后分析edge cases，然后开始写代码，最后我test了一遍，他问了我Big O和optimization，lz问了小哥的项目和组，大家很开心地结束了谈话。

thought：
给一个rows * cols大小的screen，和一组不为空的words，问这些词能再屏幕上显示几遍，一个词只能在一行，剩下的空间‘-’表示空格。

Use dynamic programming to solve this problem.
dp[index] means if the ith word start at this row, then the start word of next row'd index in sentence is dp[index].
dp[index] can be larger than the length of the sentence, in this case, one row can span multiple sentences.
**Approach 1**: Dynamic programming
code：
```java
class Solution {
    public int wordsTyping(String[] sentence, int rows, int cols) {
        int n = sentence.length;
        int[] dp = new int[n];
        int prev = 0, len = 0;
        for (int i = 0; i < n; i++){
            //when this row is full, remove previous word and space
            if (i != 0 && len > 0){
                //remove last ith word and its space, its the new length start at ith word
                len -= sentence[i - 1].length() + 1;
            }
            //calculate the length of each line and get next line index;
            //to avoid array out of bound, using %
            while (len + sentence[prev % n].length() <= cols){
                len += sentence[prev % n].length() + 1;
                prev++;
            }
            //it is if we start at ith word, next row we start at dp[i] word.
            dp[i] = prev;
        }
        int count = 0;
        for (int i = 0, k = 0; i < rows; i++){
            // count how many words one row has and move to start of next row.
            count += dp[k] - k;
            k = dp[k] % n;
        }
        //we have n words, so we could have count / n times 
        return count / n;
    }
}
```

**Approach 2**:
```java
public class Solution {
    public int wordsTyping(String[] sentence, int rows, int cols) {
        String s = String.join(" ", sentence) + " ";
        int start = 0, l = s.length();
        for (int i = 0; i < rows; i++) {
            start += cols;
            if (s.charAt(start % l) == ' ') {
                start++;
            } else {
                while (start > 0 && s.charAt((start-1) % l) != ' ') {
                    start--;
                }
            }
        }
        
        return start / s.length();
    }
}
```

13. 486.Predict the Winner (高频 3)
[LeetCode](https://leetcode.com/problems/predict-the-winner/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/84243459)

key word: DP

出现面经：

thought：
给一个数组，palyer只能从当前没取的数的最左边或者最右边取，每次拿一个，而且是最优的，问先手的人能否赢。

如果只有偶数个数，先手必胜

用一个二维数组进行dp，全赋值为-1，然后用一个helper，得到从0 -》 n-1，player1能拿到的score， 如果比总score大或等于就赢了。

Using a two dimensional array to perform dynamic programming. Fill dp with -1 first. Because Arrays.fill can't assign value to multi dimensional array, we should do it row by row. Then calculate the sum of the whole nums for future judgement.

Because when the number is a even number, first taker will always win. Because he could always leave a worse choice for the second player.

For the predict helper, the dp array stores how much score first player could make after one round. One round is that first player takes one number and second player takes one number too. Their position are I and j. If we have dp value , return it. Or calculate different circumstance and take the bigger one.

code：
```java
class Solution {
    public boolean PredictTheWinner(int[] nums) {
        if (nums.length % 2 == 0){
            return true;
        }
        
        int n = nums.length;
        int[][] dp = new int[n][n];
        for (int[] row : dp){
            Arrays.fill(row, -1);
        }
        
        int sum = 0;
        for (int i : nums){
            sum += i;
        }
        int score = predictHelper(nums, dp, 0, n - 1);
        return 2 * score >= sum;
    }
    
    private int predictHelper(int[] nums, int[][] dp, int i, int j){
        if (i > j){
            return 0;
        }
        if (dp[i][j] != -1){
            return dp[i][j];
        }
        //因为二号选手会取让你分低的，选更小的那个。
        //一个round的情况，取左，二号选手取左或者右。
        int a = nums[i] + Math.min(predictHelper(nums, dp, i + 1, j - 1),predictHelper(nums, dp, i + 2, j));
        //一个round的情况，取右，二号选手取左或者右。
        int b = nums[j] + Math.min(predictHelper(nums, dp, i, j - 2),predictHelper(nums, dp, i + 1, j - 1));
        dp[i][j] = Math.max(a, b);
        
        return dp[i][j];
    }
}
```


14. 505.The Maze II (高频 3)
[LeetCode](https://leetcode.com/problems/the-maze-ii/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/84252213)

key word: DFS, BFS

出现面经：
[Google Onsite 血跪](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=452151&ctid=84995)
迷宫. 给你一个有一些墙的迷宫. 怎么从start去到end


thought：
二维数组表示的迷宫，1是墙，碰到墙才能停下，问从start到end最小距离，不能到回-1。

用一个二维数组存储每个点到start的最短距离，相当于一个dp。然后开始DFS

This question is a variation of question 490. The maze. We just need to store the value to a dist array every time we move to a new position. After that, return dist[destination[0][destination[1]].
only one dfs search, do not need reset visited.

code：
```java
class Solution {
    private int res = Integer.MAX_VALUE;
    private int[][] dirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    public int shortestDistance(int[][] maze, int[] start, int[] destination) {
        if (maze == null || maze.length == 0 || maze[0].length == 0){
            return -1;
        }
        int[][] dist = new int[maze.length][maze[0].length];
        dist[start[0]][start[1]] = 1;
        dfs(maze, start, destination, dist);
        return dist[destination[0]][destination[1]] - 1;
    }
    
    private void dfs(int[][] maze, int[] current, int[] destination, int[][] dist){
        int x = current[0];
        int y = current[1];
        //到了返回
        if (x == destination[0] && y == destination[1]){
            return;
        }
        for (int[] dir : dirs){
            int xx = x, yy = y;
            int count = dist[x][y];
            //get 现在位置距离起点的距离，然后往一个方向走直到墙壁。
            while (xx + dir[0] >= 0 && yy + dir[1] >= 0 && xx + dir[0] < maze.length && yy + dir[1] < maze[0].length && maze[xx + dir[0]][yy + dir[1]] == 0){
                xx += dir[0];
                yy += dir[1];
                count++;
            }
            //如果这是条远路，再见
            if (dist[xx][yy] > 0 && count >= dist[xx][yy]){
                continue;
            }
            //更新dist，继续dfs。
            dist[xx][yy] = count;
            dfs(maze, new int[]{xx, yy}, destination, dist);
        }
    }
}
```


15. 642.Design Search Autocomplete System
[LeetCode](https://leetcode.com/problems/design-search-autocomplete-system/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/84265738)

key word: Trie

出现面经：

thought：
实现一个自动补全的类

We design a Trie node class which has a map to contain the next children, a counts map stores the frequency and a boolean is word which works to denote a whole sentence.

In the constructor, we get sentences and corresponding times. We new a root node and an empty prefix. Then using a for loop to add sentence and its times to our trie tree.

In the private method add, Get root first, then for this sentence s, convert it to a char array and store them into trie tree. If trie tree does not have this children, add a new trie node to the map. Move forward and store count.

In input, we want to get the top three sentences with highest times. So we need a priority queue to implement it. If c is '#' it means input is over, we add this input sentence as prefix and count = 1 into trie tree, then return an empty list. Each time call input, we add this char to prefix, then search the trie tree. If trie tree does not have this prefix, return an empyt list.

After that, we are sure this prefix is contained in trie tree. So use a priority queue to get the current node. counts map and add all its entries into pq. New an array list which is res and poll top three from pq.
code：
```java
class AutocompleteSystem {
    class TrieNode{
        Map<Character, TrieNode> children;
        Map<String, Integer> counts;
        boolean isWord;
        public TrieNode(){
            children = new HashMap<>();
            counts = new HashMap<>();
            isWord = false;
        }
    }
    
    TrieNode root;
    String prefix;

    public AutocompleteSystem(String[] sentences, int[] times) {
        root = new TrieNode();
        prefix = "";
        
        for (int i = 0; i < sentences.length; i++){
            add(sentences[i], times[i]);
        }
    }
    
    private void add(String s, int count){
        TrieNode cur = root;
        for (char c : s.toCharArray()){
            TrieNode next = cur.children.get(c);
            if (next == null){
                next = new TrieNode();
                cur.children.put(c, next);
            }
            cur = next;
            cur.counts.put(s, cur.counts.getOrDefault(s, 0) + count);
        }
        cur.isWord = true;
    }
    
    public List<String> input(char c) {
        if (c == '#'){
            add (prefix, 1);
            prefix = "";
            return new ArrayList<String>();
        }
        
        prefix = prefix + c;
        TrieNode cur = root;
        for (char ch : prefix.toCharArray()){
            TrieNode next = cur.children.get(ch);
            if (next == null){
                return new ArrayList<String>();
            }
            cur = next;
        }
        
        PriorityQueue<Map.Entry<String, Integer>> pq = new PriorityQueue<>((a, b) -> (a.getValue() == b.getValue() ? a.getKey().compareTo(b.getKey()) : b.getValue() - a.getValue()));
        for (Map.Entry<String, Integer> entry : cur.counts.entrySet()){
            pq.add(entry);
        }
        
        List<String> res = new ArrayList<>();
        for (int i = 0; i < 3 && !pq.isEmpty(); i++){
            res.add(pq.poll().getKey());
        }
        return res;
    }
}

/**
 * Your AutocompleteSystem object will be instantiated and called as such:
 * AutocompleteSystem obj = new AutocompleteSystem(sentences, times);
 * List<String> param_1 = obj.input(c);
 */
```
`AutocompleteSystem()` takes $O(k*l)$ time. We need to iterate over l sentences each of average length $k$, to create the trie for the given set of sentencessentences.

`input()` takes $O\big(p+q+mlog(m)$ time. Here, pp refers to the length of the sentence formed till now, cur_sencursen. q refers to the number of nodes in the trie considering the sentence formed till now as the root node. Again, we need to sort the listlist of length mm indicating the options available for the hot sentences, which takes $O\big(mlog(m) $time.



16. 659.Split Array into Consecutive Subsequences (高频 5)
[LeetCode](https://leetcode.com/problems/split-array-into-consecutive-subsequences/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/84273101)

key word: Heap，Greedy

出现面经：
[Google 2019 暑期实习二面+timeline](https://www.1point3acres.com/bbs/thread-462564-1-1.html)
4. 国人大哥
一个array，就问能不能把它split成多个顺子，比如3,4,5,6,7和4,5,6,7,8,9,10等等等

thought：
判断一个数组中的数，有重复，可不可以分割成至少有三个元素的递增连续子序列。

这道题让我们将数组分割成多个连续递增的子序列，注意这里可能会产生歧义，实际上应该是分割成一个或多个连续递增的子序列，因为[1,2,3,4,5]也是正确的解。这道题就用贪婪解法就可以了，我们使用两个哈希表map，第一个map用来建立数字和其出现次数之间的映射freq，第二个用来建立可以加在某个连续子序列后的数字及其可以出现的次数之间的映射need。对于第二个map，举个例子来说，就是假如有个连，[1,2,3]，那么后面可以加上4，所以就建立4的映射。这样我们首先遍历一遍数组，统计每个数字出现的频率，然后我们开始遍历数组，对于每个遍历到的数字，首先看其当前出现的次数，如果为0，则继续循环；如果need中存在这个数字的非0映射，那么表示当前的数字可以加到某个连的末尾，我们将当前数字的映射值自减1，然后将下一个连续数字的映射值加1，因为当[1,2,3]连上4后变成[1,2,3,4]之后，就可以连上5了；如果不能连到其他子序列后面，我们来看其是否可以成为新的子序列的起点，可以通过看后面两个数字的映射值是否大于0，都大于0的话，说明可以组成3连儿，于是将后面两个数字的映射值都自减1，还有由于组成了3连儿，在need中将末尾的下一位数字的映射值自增1；如果上面情况都不满足，说明该数字是单牌，只能划单儿，直接返回false。最后别忘了将当前数字的freq映射值自减1。

code：
```java
class Solution {
    public boolean isPossible(int[] nums) {
        Map<Integer, Integer> frequency = new HashMap<>();
        Map<Integer, Integer> append = new HashMap<>();
        for (int n : nums){
            frequency.put(n, frequency.getOrDefault(n, 0) + 1);
        }
        for (int n : nums){
            if (frequency.get(n) == 0){
                continue;
            }
            else if (append.getOrDefault(n, 0) > 0){
                append.put(n, append.get(n) - 1);
                append.put(n + 1, append.getOrDefault(n + 1, 0) + 1);
            }
            else if (frequency.getOrDefault(n + 1, 0) > 0 && frequency.getOrDefault(n + 2, 0) > 0){
                frequency.put(n + 1, frequency.get(n + 1) - 1);
                frequency.put(n + 2, frequency.get(n + 2) - 1);
                append.put(n + 3, append.getOrDefault(n + 3, 0) + 1);
            }
            else{
                return false;
            }
            frequency.put(n, frequency.get(n) - 1);
        }
        return true;
    }
}
```


17. 676.Implement Magic Dictionary (高频 3，变种)
[LeetCode](https://leetcode.com/problems/implement-magic-dictionary/description/)
[blog]()

key word: Hash, Trie

出现面经：
[狗家 11轮面试 3轮电话 + 5轮onsite + 3轮加面(onsite)](https://www.1point3acres.com/bbs/thread-438216-1-1.html)
1.高频题 给你一个String target 还有一个List<String> dictionary 要求你输出-----所有给target字符串 添加字符之后等于dictionary 的单词
比如 target ----> google    List<goooooogle, ddgoogle,  abcd, googles>   return List<goooooogle, ddgoogle, googles> 
变种，要求不是替换，而是添加。
ANS：
此变种可以通过hashset，然后看每一个stirng是否含有target String的所有字符。


thought：
设计一个magic dictionary，给一组词构建字典，然后search，如果能只改变一个字符就成为字典中的词，返回true。

设计一棵trie树，然后search时每次变一个字符，看能否找到在trie中的word。

**Approach 1**：Trie
Implement a trie tree which store all the strings. Then in search, change the char of search string one by and find if it is in the trie tree.

code:
```java
class MagicDictionary {
    class TrieNode{
        TrieNode[] children = new TrieNode[26];
        boolean isWord;
        public TrieNode() {}
    }
    private TrieNode root;
    /** Initialize your data structure here. */
    public MagicDictionary() {
        root = new TrieNode();
    }
    
    /** Build a dictionary through a list of words */
    public void buildDict(String[] dict) {
        for (String s : dict){
            TrieNode node = root;
            for (char ch : s.toCharArray()){
                if (node.children[ch - 'a'] == null){
                    node.children[ch - 'a'] = new TrieNode();
                }
                node = node.children[ch - 'a'];
            }
            node.isWord = true;
        }
    }
    
    /** Returns if there is any word in the trie that equals to the given word after modifying exactly one character */
    public boolean search(String word) {
        char[] arr = word.toCharArray();
        for (int i = 0; i < word.length(); i++){
            for (char ch = 'a'; ch <= 'z'; ch++){
                if (arr[i] == ch){
                    continue;
                }
                char org = arr[i];
                arr[i] = ch;
                if (searchHelper(arr)){
                    return true;
                }
                arr[i] = org;
            }
        }
        return false;
    }
    
    private boolean searchHelper(char[] arr){
        TrieNode node = root;
        for (char ch : arr){
            if (node.children[ch - 'a'] == null){
                return false;
            }
            node = node.children[ch - 'a'];
        }
        return node.isWord;
    }
}

/**
 * Your MagicDictionary object will be instantiated and called as such:
 * MagicDictionary obj = new MagicDictionary();
 * obj.buildDict(dict);
 * boolean param_2 = obj.search(word);
 */
```
用一个hashset，先把words都存进去，然后对于每一个search，改每一个位置的字符，看在hashset中存在么

**Approach 2**: HashSet
Using a hashset to contain all the strings, and then in search, change characters one by one, find wether it is in the set.
code:
```java
class MagicDictionary {
    private Set<String> set;
    /** Initialize your data structure here. */
    public MagicDictionary() {
        set = new HashSet<String>();
    }
    
    /** Build a dictionary through a list of words */
    public void buildDict(String[] dict) {
        set.clear();
        for (String str : dict){
            set.add(str);
        }
    }
    
    /** Returns if there is any word in the trie that equals to the given word after modifying exactly one character */
    public boolean search(String word) {
        StringBuilder sb = new StringBuilder(word);
        for (int i = 0; i < word.length(); i++){
            char temp = word.charAt(i);
            for (int j = 0; j < 26; j++){
                if ('a' + j == temp){
                    continue;
                }
                sb.setCharAt(i, (char)('a' + j));
                if (set.contains(sb.toString())){
                    return true;
                }
                sb.setCharAt(i, temp);
            }
        }
        return false;
    }
}

/**
 * Your MagicDictionary object will be instantiated and called as such:
 * MagicDictionary obj = new MagicDictionary();
 * obj.buildDict(dict);
 * boolean param_2 = obj.search(word);
 */
```


18. 684.Redundant Connection（高频 10）
[LeetCode](https://leetcode.com/problems/redundant-connection/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/84314575)

key word: Tree, Union find, Graph

出现面经：
[Google Onsite 血跪](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=452151&ctid=84995)
给你一个tree. 里面多了一条edge, 所以不是binary tree了. Remove 那个多余的edge.

[狗家技术电面-十二月第一波](https://www.1point3acres.com/bbs/thread-462577-1-1.html)
给了一个 n * 2 的 2D array， 里面是一个undirected graph 的所有的edges，
然后其中有一条边使得graph形成cycle。找出这条边

利口陆扒寺

[狗家11月初 昂赛](https://www.1point3acres.com/bbs/thread-457158-1-1.html)
第二问 树删redundant边


thought：
给一个无向图，N个点，labeled 1 - N（可以作为数组index）本来是没有环的，多了一条边，找到可以被删掉的多的这条边，若有多个解，返回在二维数组中出现晚的那条。

使用union-find来解决，只要出现了环，就把这条边返回。

Using union find to solve this problem. We maintain a parents array which contains the parent node of each node. Then use find() method to connect different connected component together. If one edge formats a circle, return it.
code：
```java
class Solution {
    public int[] findRedundantConnection(int[][] edges) {
        int[] parents = new int[edges.length + 1];
        for (int[] edge: edges){
            if (find(parents, edge[0]) == find(parents, edge[1])){
                return edge;
            }
            else{
                parents[find(parents, edge[0])] = find(parents, edge[1]);
            }
        }
        return new int[2];
    }
    
    private int find (int[] parents, int n){
        if (parents[n] == 0){
            return n;
        }
        else{
            return find(parents, parents[n]);
        }
    }
}
```


19. 685.Redundant Connection II
[LeetCode](https://leetcode.com/problems/redundant-connection-ii/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/84315408)

key word:Tree, DFS, Union-Find, Graph

出现面经：

thought：

同上，这波变成有向图了

Now the Graph is directed. How to find the added edge.

code：
```java
class Solution {
    public int[] findRedundantDirectedConnection(int[][] edges) {
        int[] parents = new int[edges.length + 1];
        for (int i = 0; i <= edges.length; i++){
            parents[i] = i;
        }
        
        int[] candidate1 = null, candidate2 = null;
        for (int[] edge : edges){
            int parentx = find(parents, edge[0]), parenty = find(parents, edge[1]);
            if (parentx != parenty){
                if (parenty != edge[1]){
                    candidate1 = edge;// record the last edge which results in "multiple parents
                }
                else{
                    parents[parenty] = parentx;
                }
            }
            else{
                candidate2 = edge;// record last edge which results in "cycle" issue, if any.
            }
        }
        
        if (candidate1 == null){
            return candidate2;
        }
        if (candidate2 == null){
            return candidate1;
        }
        // If both issues present, then the answer should be the first edge which results in "multiple parents" issue
        for (int[] edge : edges){
            if (edge[1] == candidate1[1]){
                return edge;
            }
        }
        return new int[2];
    }
    
    private int find(int[] parents, int n){
        while (n != parents[n]){
            parents[n] = parents[parents[n]];
            n = parents[n];
        }
        return n;
    }
}
```


20. 731.My Calendar II
[LeetCode](https://leetcode.com/problems/my-calendar-ii/description/)
[blog]()

key word: Array

出现面经：
[狗家昂赛new grad](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=450642)
第二轮：一个白人大姐，LC731，本来很简单的，但是问题是从一开始一个list of interval找彼此之间的overlap开始，我sort之后的方法错了，试了test case才发现，然后改回two pointers
这一轮给我感觉有些悬念吧

thought：
实现一个calendar类，可以重复两次，但是不能重复三次。

思路是用一个List存overlap的区间，然后再判断一次overlap

当第一book的时候，calendar为空，for循环跳过，直接在最后加上然后返回true；
第二次book的时候，calendar有interval，进去for，得到overlap的start和end，如果不合法，continue，继续加上返回true
第i次book的时候，进入for，然后同样的，对于每一个calenday里的区间，判断overlap，然后成立就返回false，不成立就加到overlap里，表示这一块儿已经double booking了。

find overlaps first, then in overlaps, find another triple booking.

code：
```java
class MyCalendarTwo {
    private List<int[]> calendar;
    public MyCalendarTwo() {
        calendar = new ArrayList<>();
    }
    
    public boolean book(int start, int end) {
        List<int[]> overlaps = new ArrayList<>();
        for (int[] slot : calendar){
            int overlapStart = Math.max(slot[0], start);
            int overlapEnd = Math.min(slot[1], end);
            if (overlapStart >= overlapEnd){
                continue;
            }
            for (int[] overlap : overlaps){
                int secondStart = Math.max(overlap[0], overlapStart);
                int secondEnd = Math.min(overlap[1], overlapEnd);
                if (secondStart < secondEnd){
                    return false;
                }
            }
            overlaps.add(new int[]{overlapStart, overlapEnd});
        }
        calendar.add(new int[]{start, end});
        return true;
    }
}

/**
 * Your MyCalendarTwo object will be instantiated and called as such:
 * MyCalendarTwo obj = new MyCalendarTwo();
 * boolean param_1 = obj.book(start,end);
 */
```


21. 750.Number Of Corner Rectangles (高频 5)
[LeetCode](https://leetcode.com/problems/number-of-corner-rectangles/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/84321184)

key word: DP

出现面经：

thought：
数组中有0和1，找到四个角由1组成的所有矩形的数量。

我们用两个for循环来先选择两行，然后统计这两行上处于同一列的点对的数量，如果要形成矩形，在点对中任意选择两组就可以了，也就是
count * （count - 1）。吧所有的结果加和起来。

Process from fix an edge first then column by column.

code：
```java
class Solution {
    public int countCornerRectangles(int[][] grid) {
        if (grid == null || grid.length <= 1){
            return 0;
        }
        int res = 0;
        for (int i = 0; i < grid.length - 1; i++){
            for (int j = i + 1; j < grid.length; j++){
                int count = 0;
                for (int k = 0; k < grid[0].length; k++){
                    if (grid[i][k] == 1 && grid[j][k] == 1){
                        count++;
                    }
                }
                res += count * (count - 1) / 2;
            }
        }
        return res;
    }
}
```


22. 769.Max Chunks To Make Sorted (高频 3)
[LeetCode](https://leetcode.com/problems/max-chunks-to-make-sorted/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/84325209)

key word:Array

出现面经：


thought：
一个n个元素的数组，是有0 - n-1个元素排列组成的，问最多能把数组分成几块，使得分别排列之后能够得到升序的数组。

用一个max数组记录每个位置到现在为止的最大值，然后再一个循环，当 i == max[i]的时候，可以分块了，因为最大值元素除非是分块能够包含它的正确位置了，它才能正确的排序。

max array is used to contain current max value from the iteration from left to right. If i == max[i], it means we could sort to here.
code：
```java
class Solution {
    public int maxChunksToSorted(int[] arr) {
        if (arr == null || arr.length == 0){
            return 0;
        }
        int[] max = new int[arr.length];
        max[0] = arr[0];
        for (int i = 1; i < arr.length; i++){
            max[i] = Math.max(max[i - 1], arr[i]);
        }
        int count = 0;
        for (int i = 0; i < arr.length; i++){
            if (max[i] == i){
                count++;
            }
        }
        return count;
    }
}
```


23. 774.Minimize Max Distance to Gas Station (高频 3)
[LeetCode](https://leetcode.com/problems/minimize-max-distance-to-gas-station/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/84332901)

key word: Binary Search

出现面经：

thought：
在一个一维数组上，每个数组元素表示加油站i的位置，我们要加入K个新的加油站，使得两个相邻加油站的最大距离最小。

二分搜索，left = 0， right为station间最大距离，然后可以找合适的最大的距离，计算出一个mid距离，再计算每一个station之间如果是这个距离的话，能插入几个新station，加起来

如果比K大，那么现在的距离太小了，left = mid；如果count <= k, 那么太大了，right = mid。


code：
```java
class Solution {
    public double minmaxGasDist(int[] stations, int K) {
        int count = 0, n = stations.length;
        double left = 0, right = stations[n - 1] - stations[0], mid = 0;
        
        while (left + 1e-6 < right){
            mid = (left + right) / 2;
            count = 0;
            for (int i = 0; i < n - 1; i++){
                count += Math.floor((stations[i + 1] - stations[i]) / mid);
            }
            if (count > K){
                left = mid;
            }
            else{
                right = mid;
            }
        }
        return left;
    }
}
```

PriorityQueue Approach
```java
class Solution {
    public double minmaxGasDist(int[] stations, int K) {

        Arrays.sort(stations);
        PriorityQueue<Interval> que = new PriorityQueue<Interval>(new Comparator<Interval>() {
            public int compare(Interval a, Interval b) {

                double diff = a.distance() - b.distance();
                if (diff < 0) { return +1; }
                else if (diff > 0) { return -1; }
                else { return 0; }
            }
        });

        double leftToRight = stations[stations.length-1] - stations[0];
        int remaining = K;

        for (int i = 0; i < stations.length-1; i++) {
            int numInsertions = (int)(K*(((double)(stations[i+1]-stations[i]))/leftToRight));
            que.add(new Interval(stations[i], stations[i+1], numInsertions));
            remaining -= numInsertions;
        }

        while (remaining > 0) {
            Interval interval = que.poll();
            interval.numInsertions++;
            que.add(interval);
            remaining--;
        }

        Interval last = que.poll();
        return last.distance();

    }

    class Interval {
        double left;
        double right;
        int numInsertions;
        double distance() { return (right - left)/  ((double)(numInsertions+1)) ; }
        Interval(double left, double right, int numInsertions) 
        { 
            this.left = left; 
            this.right = right; 
            this.numInsertions = numInsertions; 
            
        }
    }
}
```


24. 803.Bricks Falling When Hit (高频 4)
[LeetCode](https://leetcode.com/problems/bricks-falling-when-hit/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/83729912)

key word: DFS

出现面经：
[Google onsite面筋](https://www.1point3acres.com/bbs/thread-461856-1-1.html)
第二轮：有一个celling，从上面延伸下来很多brick，这些brick只有在跟有celling连接的brick相邻才可以
不掉下来。问remove掉一个brick，会跌下来的brick的数量。

thought：
类似于泡泡龙的游戏，给一个二维数组，1表示砖，只有与顶部相连的砖才是不会掉的，给一个hit的位置的二维数组，问每次打这个位置会掉下多少个砖。

第一步，把所有要remove的位置都置为0

第二步，把所有和top连着的砖都置为2，表示连着，用dfs。

第三步，按照倒序，把remove的部分加回去，然后如果它和top是连着的，也就是周围有2，那就dfs，get这次打掉了多少砖。

DFS, fisrt remove all hits, then add them back from last to first. DFS recursively get number of bricks one by one.

code：
```java
class Solution {
    private static final int TOP = 2;
    private static final int BRICK = 1;
    private static final int EMPTY = 0;
    private static final int[][] DIRS = {{1, 0}, {-1, 0}, {0, 1},{0, -1}};
    public int[] hitBricks(int[][] grid, int[][] hits) {
        int[] res = new int[hits.length];
        //remove all hits and then add them back
        for (int[] hit : hits){
            grid[hit[0]][hit[1]]--;
        }
        
        //assign 2 to all grids connected to top;
        for (int i = 0; i < grid[0].length; i++){
            dfs(0, i, grid);
        }
        
        //add back removed hitted bricks
        for (int i = hits.length - 1; i >= 0; i--){
            int x = hits[i][0], y = hits[i][1];
            grid[x][y]++;
            if (grid[x][y] == BRICK && isConnected(x, y, grid)){
                res[i] = dfs(x, y, grid) - 1;
            }
        }
        return res;
    }
    
    private int dfs(int i, int j, int[][] grid){
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] != BRICK){
            return 0;
        }
        grid[i][j] = 2;
        return dfs(i + 1, j, grid) +
        dfs(i - 1, j, grid) +
        dfs(i, j + 1, grid) +
        dfs(i, j - 1, grid) + 1;//remember itself
    }
    
    private boolean isConnected(int i, int j, int[][] grid){
        if (i == 0){//first row，connected
            return true;
        }
        for (int[] dir : DIRS){
            int x = i + dir[0], y = j + dir[1];
            if (x >= 0 && x < grid.length && y >= 0 && y < grid[0].length && grid[x][y] == TOP){
                return true;
            }
        }
        return false;
    }
}
```

25. 815.Bus Routes (高频 5)
[LeetCode](https://leetcode.com/problems/bus-routes/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/84332912)

key word:BFS

出现面经：

thought：
给一个二位数组，每一个i表示第i条 bus route，给一个开始站和结束站，问能需要坐几辆才能到达。不能-1.

如果一开始就相同，直接返回，然后建立一个bus stop到它可以坐的公交线的map，然后开始BFS。

建立方式，对于第i条公交线，上面的每一个stop j，在map中get它的线路，add(i)再重新put 回去。

BFS，queue中加入S，然后开始。 res++先，然后把S的所有公交线路都进行判断，如果到过了这条线，就continue，否则就对线路上的站点进行判断，相同就返回res，否则就压入队列中，表示我这次坐车能到的所有站。

如果BFS结束都没有找到，return -1；

code：
```java
class Solution {
    public int numBusesToDestination(int[][] routes, int S, int T) {
        //a map contains map from bus stop number to the bus routes could be switched to.
        Map<Integer, ArrayList<Integer>> map = new HashMap<>();
        //a queue contains current stops we could reach.
        Queue<Integer> queue = new LinkedList<>();
        HashSet<Integer> visited = new HashSet<>();
        int res = 0;
        
        if (S == T){
            return res;
        }
        
        for (int i = 0; i < routes.length; i++){
            for (int j = 0; j < routes[i].length; j++){
                ArrayList<Integer> buses = map.getOrDefault(routes[i][j], new ArrayList<>());
                buses.add(i);
                map.put(routes[i][j], buses);
            }
        }
        
        queue.offer(S);
        while (!queue.isEmpty()){
            int size = queue.size();
            res++;
            for (int i = 0; i < size; i++){
                int current = queue.poll();
                ArrayList<Integer> buses = map.get(current);
                for (int bus : buses){
                    if (visited.contains(bus)){
                        continue;
                    }
                    visited.add(bus);
                    for (int j = 0; j < routes[bus].length; j++){
                        if (routes[bus][j] == T){
                            return res;
                        }
                        queue.offer(routes[bus][j]);
                    }
                }
            }
        }
        return -1;
    }
}
```

26. 834.Sum of Distances in Tree (高频 3)
[LeetCode](https://leetcode.com/problems/sum-of-distances-in-tree/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/84337135)

key word: DFS

出现面经：
[狗家11月初 昂赛](https://www.1point3acres.com/bbs/thread-457158-1-1.html)
1. 利口 把三思 变种 找出最短dist的某一个node返回

thought：
给一个无向图，是一棵树，labeled from 0 to N - 1.问从每个点开始，到其他所有点的距离的和，每一跳算距离 1.

用一个List<HashSet<Integer>>来储存图。list 的index就是 索引。第一个循环add HashSet<>()，第二个循环get[0]。add[1], get[1].add[0];

然后两个dfs存储距离结果，返回全局res。
count[i] 表示i子树拥有的nodes数，res[i]表示i子树的距离和。

0. Let's solve it with node 0 as root.

1. Initial an array of hashset tree, tree[i] contains all connected nodes to i.
Initial an array count, count[i] counts all nodes in the subtree i.
Initial an array of res, res[i] counts sum of distance in subtree i.

2. Post order dfs traversal, update count and res:
count[root] = sum(count[i]) + 1
res[root] = sum(res[i]) + sum(count[i])

3. Pre order dfs traversal, update res:
When we move our root from parent to its child i, count[i] points get 1 closer to root, n - count[i] nodes get 1 futhur to root.
res[i] = res[root] - count[i] + N - count[i]

4. return res, done.

code：
```java
class Solution {
    private int[] res;
    private int[] count;
    private ArrayList<HashSet<Integer>> tree;
    private int n;
    public int[] sumOfDistancesInTree(int N, int[][] edges) {
        tree = new ArrayList<>();
        res = new int[N];
        count = new int[N];
        n = N;
        for (int i = 0; i < N; i++){
            tree.add(new HashSet<Integer>());
        }
        for (int[] edge : edges){
            tree.get(edge[0]).add(edge[1]);
            tree.get(edge[1]).add(edge[0]);
        }
        dfs(0, new HashSet<Integer>());
        dfs2(0, new HashSet<Integer>());
        return res;
    }
    
    private void dfs(int root, HashSet<Integer> seen){
        seen.add(root);
        for (int i : tree.get(root)){
            if (!seen.contains(i)){
                dfs(i, seen);
                count[root] += count[i];
                res[root] += res[i] + count[i];
            }
        }
        count[root]++;
    }
    
    private void dfs2(int root, HashSet<Integer> seen){
        seen.add(root);
        for (int i : tree.get(root)){
            if (!seen.contains(i)){
                res[i] = res[root] - count[i] + n - count[i];
                dfs2(i, seen);
            }
        }
    }
}
```

27. 489.Robot Room Cleaner (高频 7)
[LeetCode](https://leetcode.com/problems/robot-room-cleaner/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/83729710)

key word: DFS

出现面经：
[谷歌 过经](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=446929)
巴西人肤色的姐姐，白小哥shadow。问我个高频题，扫地机器人，利口死罢就。一开始是长方形地图，秒了之后改成不规则地图，又秒了。还剩下15分钟，尬聊，问了好多组里的事。

[MV苟昂塞 ](https://www.1point3acres.com/bbs/thread-446364-1-1.html?_dsign=440dd36a)
第一轮是印度大叔，口音比较重，扫地机器人，地里不少，就不详细讲啦。


thought：
给一个二维数组中的robot cleaner。它有4个API，move()向前，如果不能走返回false；turnLeft()左转90度，turnRihgt（）右转90读，clean（）清。给一方法，让robot clean each cells
```java
interface Robot {
  // returns true if next cell is open and robot moves into the cell.
  // returns false if next cell is obstacle and robot stays on the current cell.
  boolean move();

  // Robot will stay on the same cell after calling turnLeft/turnRight.
  // Each turn will be 90 degrees.
  void turnLeft();
  void turnRight();

  // Clean the current cell.
  void clean();
}
```

backtrack的思想解决，实际上就是DFS。

对于backtrack方法，记录现在的位置，方向和一个visited 的set String or whatever
一开始，先判断是不是访问过，访问过直接返回。把这个点加进visited，clean

然后进for循环，前进4次。如果能前进，根据现在的方向，确定下一个格子的位置，然后递归backtrack。结束之后，转向，move，转向，回到原来的位置和方向，if 结束。

结束后，转向，记得要%360.

因为不需要回退，所以不用remove。

code：
```java
/**
 * // This is the robot's control interface.
 * // You should not implement it, or speculate about its implementation
 * interface Robot {
 *     // Returns true if the cell in front is open and robot moves into the cell.
 *     // Returns false if the cell in front is blocked and robot stays in the current cell.
 *     public boolean move();
 *
 *     // Robot will stay in the same cell after calling turnLeft/turnRight.
 *     // Each turn will be 90 degrees.
 *     public void turnLeft();
 *     public void turnRight();
 *
 *     // Clean the current cell.
 *     public void clean();
 * }
 */
class Solution {
    public void cleanRoom(Robot robot) {
        Set<String> set = new HashSet<>();
        backtrack(robot, set, 0, 0, 0);
    }
    
    private void backtrack(Robot robot, Set<String> set, int i, int j, int curDirection){
        String temp = i + "," + j;
        if (set.contains(temp)){
            return;
        }
        
        robot.clean();
        set.add(temp);
        
        for (int n = 0; n < 4; n++){
            if (robot.move()){
                int x = i, y = j;
                switch(curDirection){
                    case 0:{
                        x = i - 1;
                        break;
                    }
                    case 90:{
                        y = j + 1;
                        break;
                    }
                    case 180:{
                        x = i + 1;
                        break;
                    }
                    case 270:{
                        y = j - 1;
                        break;
                    }
                    default:{
                        break;
                    }
                }
                backtrack(robot, set, x, y, curDirection);
                //go back to orginal position
                robot.turnLeft();
                robot.turnLeft();
                robot.move();
                robot.turnRight();
                robot.turnRight();
            }
            //turn to next direction
            robot.turnRight();
            curDirection += 90;
            curDirection %= 360;
        }
    }
}
```

28. 837.New 21 Game
[LeetCode](https://leetcode.com/problems/new-21-game/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/84338606)

key word: DP

出现面经：
[ Google Onsite 血跪](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=452151&ctid=84995)

[狗家10/31Sunnyvale面经](https://www.1point3acres.com/bbs/thread-458802-1-1.html)
3. 一个类似21点的游戏，假设牌桌上有无数张1-10的牌，然后你手上的牌的总和是k，现在你可以随机到牌桌上抽牌加到总和里，如果你手上牌的总和在20-25之间就是win，如果总和超过25就是lose，现在让你求lose的概率。
这一题我好像在地里见到过但是当时那个楼主说他没有做出来，所以我就也附上一下我的大概解法。
首先因为每张牌都是无数张，所以抽任何一张牌的概率都是0.1。然后就是要考虑到有很多重复的情况，所以用dp或者recursion with memoization其实都可以。
我是用dp做的，从后往前推，所有的结束可能就是20 - 29 其中P(20)到P(25)= 0， P(26)到P(29) = 1。那么P(19) = 0.1*P(20) + 0.1*P(21)+.... 以此类推，最后算到P(k)
followup：假设每张牌是n张
这就比较麻烦了，因为抽牌的概率跟当前牌桌上每张牌的数量有关，所以用dp比较难做，我就改用recursion with memoization。不仅要存手上牌的总和还要存牌桌上每张牌的数量。

thought：
通过画图来推理。
从0分开始， 从1 - W直接抽牌，概率相同，抽多次，大于等于K了就停止抽牌；求分数 大于等于K， 小于等于N的概率。

如果K == 0，不让抽；或者 K - 1 + W <= N，因为抽到K就停了，所以我们能得到的最大数就是K - 1 + W，这都不行也肯定是1.

我们用一个double的 dp[] 表示得到i分的概率。dp[0] = 1。wSum是能得到这个数的权重和， wSum / W就是得到这个数的概率。

dp[i] = wSum / W, 在到达K之前，Wsum + dp[i]，因为i可以作为得到后面数的跳板，如果超过K，开始记录加和。如果超过了W的window size限制，需要去掉之前的dp[i - W]的值在wSum中。

It is a conditional probability problem. To get the final answer we should calculate the before probability to get point i. If K == 0 or N >= K - 1 + W, Definitely it won't be busted. Then we could maintain a dp array, dp[i] is the probability that get point i. Wsum is the current probability to get i. It is maintained by a window with size W. If is out of window, remove the first probability.

code：
```java
class Solution {
    public double new21Game(int N, int K, int W) {
        if (K == 0 || N >= K - 1 + W){
            return 1.0;
        }
        //dp[i]: probability of get points i. dp[i] = sum(last W dp values) / W
        double[] dp = new double[N + 1];
        double wSum = 1, res = 0;
        dp[0] = 1;
        for (int i = 1; i <= N; i++){
            dp[i] = wSum / W;
            if (i < K){
                wSum += dp[i];
            }
            else{
                res += dp[i];
            }
            if (i - W >= 0){
                wSum -= dp[i - W];
            }
        }
        return res;
    }
}
```

29. 843.Guess the Word (高频 9)
[LeetCode](https://leetcode.com/problems/guess-the-word/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/83655180)

出现面经：
[Google Onsite 血跪](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=452151&ctid=84995)
猜字游戏. 给你一个secret word, 一个guess word. Return `<number of words that is guessed correctly and at correct position, number of words that is guessed correctly but at wrong position>`

[狗狗家阳谷县Onsite](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=461671&extra=page%3D1%26filter%3Dsortid%26sortid%3D311%26sortid%3D311)
第三轮 先聊简历，然后是高频猜字题

key word:

thought：
10次猜出指定的词。

 首先random选一个词，然后看有几个一样的，循环现在的wordlist，选出match数和现在一样的词，然后重新放入wordlist中，循环十次。

match() is a function which returns number of character match.

code：
```java
/**
 * // This is the Master's API interface.
 * // You should not implement it, or speculate about its implementation
 * interface Master {
 *     public int guess(String word) {}
 * }
 */
class Solution {
    public void findSecretWord(String[] wordlist, Master master) {
        for (int i = 0, m = 0; i < 10 && m < 6; i++){
            String guess = wordlist[new Random().nextInt(wordlist.length)];
            m = master.guess(guess);
            List<String> newWordlist = new ArrayList<>();
            for (String w : wordlist){
                if (matches(w, guess) == m){
                    newWordlist.add(w);
                }
            }
            wordlist = newWordlist.toArray(new String[newWordlist.size()]);
        }     
    }
    
    private int matches(String a, String b){
        int count = 0;
        for (int i = 0; i < a.length(); i++){
            if (a.charAt(i) == b.charAt(i)){
                count++;
            }
        }
        return count;
    }
}
```

30. 844.Backspace String Compare (高频 3)
[LeetCode](https://leetcode.com/problems/backspace-string-compare/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/83247684)

key word:

出现面经：
[Google 电面 过经](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=468744&extra=page%3D1%26filter%3Dsortid%26sortid%3D311%26searchoption%5B3086%5D%5Bvalue%5D%3D9%26searchoption%5B3086%5D%5Btype%5D%3Dradio%26searchoption%5B3088%5D%5Bvalue%5D%3D1%26searchoption%5B3088%5D%5Btype%5D%3Dradio%26searchoption%5B3046%5D%5Bvalue%5D%3D1%26searchoption%5B3046%5D%5Btype%5D%3Dradio%26searchoption%5B3109%5D%5Bvalue%5D%3D1%26searchoption%5B3109%5D%5Btype%5D%3Dradio%26sortid%3D311%26orderby%3Ddateline)

thought：
给两个String，#表示删掉前面的词，问最后两个String是否是相同的。

从两个String的后面找起，首先在没到头，并且有count数的# 或者虽然没有，但是有#的情况下，i--， j--

然后对比目前的非# char是否相同，相同就前进继续，不同就判断是否都完了，返回。

code：
```java
class Solution {
    public boolean backspaceCompare(String S, String T) {
        int i = S.length() - 1;
        int j = T.length() - 1;
        while(true){
            for (int count = 0; i >= 0 && (count > 0 || S.charAt(i) == '#'); i--){
                count += S.charAt(i) == '#' ? 1 : -1;
            }
            for (int count = 0; j >= 0 && (count > 0 || T.charAt(j) == '#'); j--){
                count += T.charAt(j) == '#' ? 1 : -1;
            }
            if (i >= 0 && j >= 0 && S.charAt(i) == T.charAt(j)){
                i--;
                j--;
            }
            else{
                return i == -1 && j == -1;
            }
        }
    }
}
```

31. 846.Hand of Straights
[LeetCode](https://leetcode.com/problems/hand-of-straights/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/84340676)

key word:

出现面经：
记得看到过找顺子的题目

thought：
手中有hand[] 牌，能否把手中的牌分成几组，每一组有W张连续的牌。

首先用一个TreeMap统计每一个数出现的频率，并且完成了数字的排序

然后用一个Queue储存从某个数开始的顺子的数量，lastChecked 为上一个检查过的数，opened表示现在我们有几个顺子正在构建，也就是下一个数，至少需要出现几次。

开始循环tree的key，首先判断是否非法了，如果现在有顺子组，而且数不连续了 it ！= lastChecked ，或者 opened > map.get(it)，这个数不足以满足顺子数，return false

往start queue中加入（map.get(it) - opened），表示从这个数开始的顺子数量。更新lastChecked，opended
如果start size满足W了，从这开始的数要减去 W 之前开始的数。

最后判断opened是不是0.也就是能够一个不剩的都分好。

code：
```java
class Solution {
    public boolean isNStraightHand(int[] hand, int W) {
        if (hand.length % W != 0){
            return false;
        }
        Map<Integer, Integer> map = new TreeMap<>();
        for (int card : hand){
            map.put(card, map.getOrDefault(card, 0) + 1);
        }
        Queue<Integer> start = new LinkedList<>();
        //lastChecked is the last number we check to add to group
        //opended is current we have opened number of group to construct.
        int lastChecked = -1, opened = 0;
        for (int it : map.keySet()){
            if ((opened > 0 && it > lastChecked + 1) || opened > map.get(it)){
                return false;
            }
            start.add(map.get(it) - opened);
            lastChecked = it;
            opened = map.get(it);
            if (start.size() == W){
                opened -= start.remove();
            }
        }
        return opened == 0;
    }
}
```

32. 849. Maximize Distance to Closest Person (高频 3)
[LeetCode](https://leetcode.com/problems/maximize-distance-to-closest-person/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/84342891)

key word:

出现面经：
这好像是真正的公园长凳，exam room 过于复杂，每次都要坐到离两边尽量远的位置。

thought：
一位数组表示的作为，0表示空，1表示有人，想要再坐下一人，使得他到离他最近的人距离最大，返回最大距离。

分成三段来考虑，用i表示上一个人的位置，j表示现在指针

第一段，当j第一次遇到人，res = Math.max(res, j); 这是坐在最左边的结果， 用i = j + 1记录下上个人位置
第二段，当i存在，j碰到了下一个人，res = Math.max(res, (j - i + 1) / 2); 这是坐在i， j中间的结果。 用i = j + 1记录下上个人位置 j - i 就是两个人之间空间的大小， 再加一是延伸的结果，然后除 2.
第三段，当j循环完了，res = Math.max(res, n - i); 坐在最右边的结果。



code：
```java
class Solution {
    public int maxDistToClosest(int[] seats) {
        int i = 0, j = 0, res = 0;
        int n = seats.length;
        for (; j < n; j++){
            if (seats[j] == 1){
                if (i == 0){
                    res = Math.max(res, j);
                }
                else{
                    res = Math.max(res, (j - i + 1) / 2);
                }
                i = j + 1;
            }
        }
        res = Math.max(res, n - i);
        return res;
    }
}
```

33. 853.Car Fleet (高频 5)
[LeetCode](https://leetcode.com/problems/car-fleet/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/84344298)

key word:

出现面经：
印象中不少啊，忘记整理了

thought：
N辆车形式在高速上，target表示终点有多远，postion表示它们开始的位置，speed表示它们的速度，当快的车碰上了满的车，会形成一个fleet，因为没发超过。
问最后会有多少个fleet通过终点。

我们把终点想成原点，用TreeMap建立一个-position[i]（这样越靠近终点的越靠前）到他到达终点的时间 (double)(target - position[i]) / speed[i] 的映射。

然后循环map.values()，此时因为它们的key是有序的了，离终点近的在前面，然后统计它们到终点的时间，如果比现在的时间time慢，说明会阻塞后面的车，形成一个fleet，res++。

注意得到keyset，entryset是 .keySet()   .entrySet()， 但是得到values是 .values()。两map都是这样的。

code：
```java
class Solution {
    public int carFleet(int target, int[] position, int[] speed) {
        Map<Integer, Double> map = new TreeMap<>();
        for (int i = 0; i < position.length; i++){
            map.put(-position[i], (double)(target - position[i]) / speed[i]);
        }
        int res = 0;
        double cur = 0;
        for (double time : map.values()){
            if (time > cur){
                cur = time;
                res++;
            }
        }
        return res;
    }
}
```

34. 855.Exam Room (高频 6)
[LeetCode](https://leetcode.com/problems/exam-room/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/83990350)

key word:

出现面经：
[骨骼电面暑期2019](https://www.1point3acres.com/bbs/thread-462295-1-1.html)
2. 有一张长椅，给长度为d， 假设现在坐着两个人， 第三个人来了怎么坐能离两人都尽可能远。 接下来有N个人依次过来呢？  假设长椅非常长，已经做了100万人， 接下来依次有人来选座位， 怎么改算法更快？ 最优还是次优？ 

额题目有个条件我忘说了，就是求得是离你所坐位置左右邻居的距离最远。 人少的条件下，其实就是考虑几种可能性，比如现在有两个人坐着。 第三个来的人要么坐在两个人的当中， 要么挑长椅最最左边一端，要么最最右边一端，取决于那种能离两个人都尽可能远。 但如果人非常多的话，并且不断源源不断有新的人坐上来， 每次都这样一个个查两两之间的间距就很复杂了。 如果要最优的话我考虑了一开始建一个max heap， 存的是所有两两之间的间距，这样每次来一个新的人，找出max的那个interval， 坐在当中，然后把这个人和左右邻居的interval也放进heap。 还有就是，面试官提示我可以把长椅分段，用distributed computing去算，可能会漏掉最优的解

thought：
考试的时候，一行有N个作为，标为 0 - N-1. 当有学生进来的时候，他要坐在尽量里旁边的人最远的地方。没人就坐在0位置，实现一个人 ExamRoom类，每次 seat 能把人安排在离两边最远的位置； 每次 leave 能离开座位，空出这个位置。

自定义一个Interval类，有x，左侧人位置，y，右侧人位置，dist，表示再坐一个人到两人之间的距离。两个特殊情况，x == -1，最左边没坐人，dist = y；否则 y == N,右边没坐人，dist = N - 1 - x；如果都合法的有人 this.dist = Math.abs(y - x) / 2;

在构造函数中，用一个Interval的priority queue，先比较dist，如果相同就返回 距离最大，如果距离相同就 返回最左边的。
加入fake new Interval(-1, N);

seat()的时候，poll出一个interval，如果 x == -1，seat 在 0 上；else if y == N，seat在N上，否则seat 在 (x + y)/2 上。 讲两端interval 插入pq。

remove(p)时，pq转为list，for 循环，找到p开始的和p结束的两个Interval，找到了就break。将两个Interval从pq中remove，然后加一个新的进去。(head.x tail.y)

code：
```java
class ExamRoom {
    private PriorityQueue<Interval> pq;
    private int N;
    
    class Interval{
        int x;//左边人的位置
        int y;//右边人的位置
        int dist;
        public Interval(int x, int y){
            this.x = x;
            this.y = y;
            if (x == -1){
                this.dist = y;
            }
            else if (y == N){
                this.dist = N - 1 - x;
            }
            else{
                this.dist = Math.abs(y - x) / 2;
            }
        }
    }
    public ExamRoom(int N) {
        this.pq = new PriorityQueue<Interval>((a, b) -> a.dist != b.dist ? b.dist - a.dist : a.x - b.x);
        this.N = N;
        pq.add(new Interval(-1, N));
    }
    
    public int seat() {
        int seat = 0;
        Interval in = pq.poll();
        if (in.x == -1){
            seat = 0;
        }
        else if(in.y == N){
            seat = N - 1;
        }
        else{
            seat = (in.x + in.y) / 2;
        }
        pq.offer(new Interval(in.x, seat));
        pq.offer(new Interval(seat, in.y));
        
        return seat;
    }
    
    public void leave(int p) {
        Interval head = null, tail = null;
        List<Interval> list = new ArrayList<>(pq);
        for (Interval in : list){
            if (in.x == p){
                tail = in;
            }
            if (in.y == p){
                head = in;
            }
            if (head != null && tail != null){
                break;
            }
        }
        pq.remove(head);
        pq.remove(tail);
        pq.offer(new Interval(head.x, tail.y));
    }
}

/**
 * Your ExamRoom object will be instantiated and called as such:
 * ExamRoom obj = new ExamRoom(N);
 * int param_1 = obj.seat();
 * obj.leave(p);
 */
```

35. 857.Minimum Cost to Hire K Workers (高频 5)
[LeetCode](https://leetcode.com/problems/minimum-cost-to-hire-k-workers/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/83654168)
出现面经：
[Google Onsite 血跪](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=452151&ctid=84995)


面经描述：
给你N幅画. 每幅画都有price和quality. 用最低价格买K幅画, with 0<K<=N
Constraint: 买的画必须遵守minimum price for 那幅画. 用了一个ratio来买一幅画, 剩余的K-1幅画都要遵守这个ratio.

thought:
有n个工人，第i个有quality[i]的工作质量，和wage[i]的最低工资要求，我们要找K个人，在满足它们最低工资要求情况下，的最低工资是多少。

用一个二维数组储存每个工人的 wage / quality 和 quality， 这样用quality一乘就可以得到应该付的wage了

讲数组按照 wage / quality从低到高排序， 用一个qSum来表示收集到的工人quality总值，这样乘合适的系数就是总工资。用一个 pq来找到最大的quality，如果window size 满了就pop它，所以要存负值。 我们的目标是最小的 wage / quality的基础上，最小的 qSum。类似于 Greedy

qSum + quality，pq + -quality，如果满了， poll， qSum + poll； 若等于K了，计算总工资。


code：
```java
class Solution {
    public double mincostToHireWorkers(int[] quality, int[] wage, int K) {
        double[][] workers = new double[quality.length][2];
        for (int i = 0; i < quality.length; ++i){
            workers[i] = new double[]{(double)(wage[i]) / quality[i], (double)quality[i]};
        }
        Arrays.sort(workers, (a, b) -> Double.compare(a[0], b[0]));
        double res = Double.MAX_VALUE, qSum = 0;
        PriorityQueue<Double> pq = new PriorityQueue<>();
        for (double[] worker : workers){
            qSum += worker[1];
            pq.add(-worker[1]);
            if (pq.size() > K){
                qSum += pq.poll();
            }
            if (pq.size() == K){
                res = Math.min(res, qSum * worker[0]);
            }
        }
        return res;
    }
}
```

36. 890. Find and Replace Pattern (高频 8)
[LeetCode](https://leetcode.com/problems/find-and-replace-pattern/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/84344746)

key word:

出现面经：
也出现过，没有整理啊

thought：
有一组word 和一个pattern word，找到words中所有match这个pattern的词。

用一个method find（）找到用 int[] 表示的这个word的pattern.
用一个map记录每个char，和他第一次出现时map的size，也就是它的pattern 序号， 0 ， 1， 2， 3这样
对于word的toCharArray()，如果map中没有，put进去序号，然后在res数组中，放上他的序号。返回

对于每一个words中的word，如果它们的pattern 和找到的一样，add进res list。

code：
```java
class Solution {
    public List<String> findAndReplacePattern(String[] words, String pattern) {
        int[] p = find(pattern);
        List<String> res = new ArrayList<>();
        for (String word : words){
            if (Arrays.equals(find(word), p)){
                res.add(word);
            }
        }
        return res;
    }
    
    private int[] find(String word){
        HashMap<Character, Integer> map = new HashMap<>();
        int n = word.length();
        int[] res = new int[n];
        for (int i = 0; i < n; i++){
            if (!map.containsKey(word.charAt(i))){
                map.put(word.charAt(i), map.size());
            }
            res[i] = map.get(word.charAt(i));
        }
        return res;
    }
}
```

37. 50.Pow(x, n)
[LeetCode](https://leetcode.com/problems/powx-n/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/79811523)
出现面经：
[Google Onsite 血跪](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=452151&ctid=84995)

面经描述：
Implement power(x,y) st it returns x^y. Optimize 它

thought:

常规做法是连乘循环，可以提升到logn，每次将n减少一半。这样偶数的话直接乘起来。Else if it is a negative number, we should multiple a 1/x, else we should multiple an x.
code：
```java
class Solution {
    public double myPow(double x, int n) {
        if (n == 0){
            return 1;
        }
        double res = myPow(x, n / 2);
        if (n % 2 == 0){
            return res * res;
        }
        else if (n < 0){
            return res * res * (1 / x);
        }
        else{
            return res * res * x;
        }
    }
}
```

38. 307.Range Sum Query - Mutable
[LeetCode](https://leetcode.com/problems/range-sum-query-mutable/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/84539909)

出现面经：
[Google Onsite 血跪](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=452151&ctid=84995)
给你一个infinite size length array, 和list of queries in the format of <i,j>, find the maximum number from index i to index j in array.

thought: 
给一个数组，要求能够返回数组中i 到 j的sum，并且更新数组的值。使用线段树 segment tree 解决。

首先设计SegmentTreeNode，有start， end表示它表示的那一段的范围，left，right是左右子树，sum或其他是它这一段的某个特性
new 的时候，记录start，end， left right 都先null， sum 为 0

全局变量 root

当创建NumArray的时候，递归build一棵segment tree，如果 start > end了，非法 返回
else 用start end 创建一个node，如果start == end， sum 就是nums[start]，此时只有一个元素，
    否则，计算它们的mid，分别递归调用buildTree（start，mid），（mid + 1， end），该node的sum 也是 左右子树的和。

当update的时候，用一个helper完成，递归完成，类似二分查找，如果root。start == end，说明找了，更新，return；
如果不是，计算mid，根据mid pos 关系去左右子树更新 ！！ 注意更新子节点，父亲的sum也要更新。

当search range的时候，依然一个helper去完成， root。start == i && root。end == j，直接返回root。sum
否则 就根据 i j 情况去递归search。 计算mid ，end 小于mid 搜左边，要不就是右边，要不就分开找。

Implement a data structure segment tree node, Which is actually a binary tree, but it could store sum/max/min/mean value of a interval in an array. So the tree node have start and end which is the range in nums array. Left and right node is like binary tree. And most important, sum or whatever value denotes a feature of this interval.

code：
```java
class NumArray {

    class SegmentTreeNode {
        int start;
        int end;
        SegmentTreeNode left;
        SegmentTreeNode right;
        int sum;
        
        public SegmentTreeNode(int start, int end){
            this.start = start;
            this.end = end;
            this.left = null;
            this.right = null;
            this.sum = 0;
        }
    }
    
    private SegmentTreeNode root = null;

    public NumArray(int[] nums) {
        root = buildTree(nums, 0, nums.length - 1);
    }
    
    private SegmentTreeNode buildTree(int[] nums, int start, int end){
        if (start > end){
            return null;
        }
        else{
            SegmentTreeNode res = new SegmentTreeNode(start, end);
            if (start == end){
                res.sum = nums[start];
            }
            else{
                int mid = start + (end - start) / 2;
                res.left = buildTree(nums, start, mid);
                res.right = buildTree(nums,mid + 1, end);
                res.sum = res.left.sum + res.right.sum;
            }
            return res;
        }
    }
    
    public void update(int i, int val) {
        updateHelper(root, i, val);
    }
    
    private void updateHelper(SegmentTreeNode root, int pos, int val){
        if (root.start == root.end){
            root.sum = val;
        }
        else{
            int mid = root.start + (root.end - root.start) / 2;
            if (pos <= mid){
                updateHelper(root.left, pos, val);
            }
            else{
                updateHelper(root.right, pos, val);
            }
            //remember update root's sum value;
            root.sum = root.left.sum + root.right.sum;
        }
    }
    
    public int sumRange(int i, int j) {
        return sumRangeHelper(root, i, j);
    }
    
    private int sumRangeHelper(SegmentTreeNode root, int start, int end){
        if (root.end == end && root.start == start){
            return root.sum;
        }
        else{
            int mid = root.start + (root.end - root.start) / 2;
            if (end <= mid){
                return sumRangeHelper(root.left, start, end);
            }
            //make sure the boundary, we divide mid + 1 to right subtree.
            else if (start >= mid + 1){
                return sumRangeHelper(root.right, start, end);
            }
            else{
                return sumRangeHelper(root.left, start, mid) + sumRangeHelper(root.right, mid + 1, end);
            }
        }
    }
}

/**
 * Your NumArray object will be instantiated and called as such:
 * NumArray obj = new NumArray(nums);
 * obj.update(i,val);
 * int param_2 = obj.sumRange(i,j);
 */
```


39. 215.Kth Largest Element in an Array (高频 3)
[LeetCode](https://leetcode.com/problems/kth-largest-element-in-an-array/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/80379590)
出现面经：
[狗家实习店面背靠背](https://www.1point3acres.com/bbs/thread-461398-1-1.html)
狗家背靠背店面：11月13日第一轮：（听不出口音，貌似国人大哥）
首先要求实现快排里的partition函数，给的输入为数组，begin index和end index，实现partition （这里楼主似乎有bug，之前一直没注意过）
然后写个测试函数测试partition
然后用partition实现find K largest element，问了时间复杂度O(n)。要求推导证明
具体请看力特口德而无药第二个discussion内容

第三种解法.

[新鲜半小时前google onsite interview MTV](https://www.1point3acres.com/bbs/thread-461342-1-1.html?_dsign=a3c606b8)
白人大叔 10+年
超高频 Get K largest element in array. 分类讨论了单list多call和单list单call的情况。多call用sort，单call用quick select，没让写代码，讲思路。装萌新先给了heap solution再到最优


thought:
找到数组中第K大的数。

方法一，直接sort整个数组，然后找 nums.length - k 个元素。
**Approach 1** Just sort the array and return the kth largest value.
code：
```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        Arrays.sort(nums);
        return nums[nums.length - k];
    }
}
```
O(nlogn) time and O(1) memory


方法二，用pq，相当于排序了，不过排序只需要 logk
**Approach 2** Maintain a priority queue
code：
```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int n : nums){
            pq.offer(n);
            if (pq.size() > k){
                pq.poll();
            }
        }
        
        return pq.peek();
    }
}
```
O(nlogk) time and O(k) memory


方法三，类似于快排，分割交换
首先更新k = length - k，这是我要找到的kth 在nums中的index
然后开始，left right，范围进行partition，如果比k小，left = k + 1，继续，反之亦然。等于了就break，返回

对于partition()，两个指针 i， j 我们选定 i位置为基准元素，当他俩在不越界的情况下，左边在小于nums[left]情况下++，右边在大于的情况下--，如果 i >= j 了就break，否则两个数swap； 整个循环结束之后，swap left， j，返回j，这就是分割的位置。

swap简单的。
**Approach 3**  Use the selection algorithm, kinds like quick sort.
code：
```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        k = nums.length - k;
        int left = 0, right = nums.length - 1;
        while (left < right){
            int j = partition(nums, left, right);
            if (j < k){
                left = j + 1;
            }
            else if (j > k){
                right = j - 1;
            }
            else{
                break;
            }
        }
            return nums[k];
    }
    
    private int partition(int[] nums, int left, int right){
        int i = left, j = right + 1;//because we use ++i，so start at left; --j,so start at right + 1.
        while (true){
            while (i < right && nums[++i] < nums[left]);
            while (j > left && nums[--j] > nums[left]);
            if (i >= j){
                break;
            }
            swap(nums, i, j);
        }
        swap(nums, left, j);
        return j;
    }
    
    private void swap(int[] nums, int i, int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```
O(N) best case / O(N^2) worst case running time + O(1) memory


40. 200.Number of Islands
[LeetCode](https://leetcode.com/problems/number-of-islands/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/79847643)
出现面经：
[狗家电面两面](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=461060&extra=page%3D2)
第一面很简单，利口两败，1和0换成圈圈叉叉，经典面筋。

thought:
给二维数组，问有多少个岛屿。dfs，遇到了就置‘0’防止下次遇到。

简单的dfs，当是1的时候就 dfs把所有连接的都变成0， count++

dfs按照传统，先查越界，不是能走的，visited（这题不需要），return
然后对本格子操作，置0
然后四个方向调用dfs四次。

code：
```java
class Solution {
    private int[][] dirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    public int numIslands(char[][] grid) {
        int count = 0;
        for (int i = 0; i < grid.length; i++){
            for (int j = 0; j < grid[i].length; j++){
                if (grid[i][j] == '1'){
                    count++;
                    dfs(grid, i, j);
                }
            }
        }
        return count;
    }
    
    private void dfs(char[][] grid, int i, int j){
        if (i < 0 || j < 0 || i >= grid.length || j >= grid[i].length || grid[i][j] != '1'){
            return;
        }
        grid[i][j] = '0';
        for (int[] dir : dirs){
            int x = i + dir[0];
            int y = j + dir[1];
            dfs(grid, x, y);
        }
    }
}
```
41. Game of Life
[LeetCode](https://leetcode.com/problems/game-of-life/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/80154692)
出现面经：
[狗家onsite](https://www.1point3acres.com/bbs/thread-460984-1-1.html)

面经描述：


thought:
Game of life，给一m×n的board，每一个位置代表一个cell，有起始状态活着1和死了0，可以与周围的8个邻居进行互动。board的下一个状态由下面的条件确定：

活着的cell，如果有少于两个活着的邻居，下一个状态会死。人口太少。
活着的cell，如果有两个或三个活着的邻居，下一个状态会继续活着。正常状态。
活着的cell，如果有多于三个活着的邻居，下一个状态会死亡。人口过多。
死了的cell，如果有正好三个活着的邻居，下一个状态会复活。人口繁衍。
计算board的下一个状态。

follow up：

in-place实现。
正常状态下board应该是无限的，如何处理二维数组的边界。
我们拓展一下，用两个bits存储每一个cell的变化

[2nd bit, 1st bit] = [next state, current state]
00 dead (next) <- dead (current)
01 dead (next) <- live (current)
10 live (next) <- dead (current)
11 live (next) <- live (current)
一开始所有的cell都是00或者01状态，也就是0和1。如果cell能够活下去，就将它变成1*的形式，那么就是0和3个活着的邻居和1和2或3个活着的邻居的情况。我们在第一遍循环的时候只需要对这两种情况进行处理。

然后第二次循环将所有的值右移一位，就是最终结果了，那些活不了的终究活不了。精髓在于：在cell中存储下一个state的状态而不改变原来的state，需要想象，变化是一瞬间同时发生的。

然后是对活着的邻居的统计，对于边界情况，循环时进行判断，不要出现越界的情况。然后是很重要的如何利用已经改变了的数据，很简单，使用board[i][j] & 1即可判断这个cell现在的状态是死是活。

code：
```java
class Solution {
    public void gameOfLife(int[][] board) {
        if (board == null || board.length == 0 || board[0].length == 0){
            return;
        }
        int row = board.length - 1;
        int col = board[0].length - 1;
        for (int i = 0; i <= row; i++){
            for (int j = 0; j <= col; j++){
                int lives = countNeighbours(board, row, col, i, j);
                if (board[i][j] == 1 && lives >= 2 && lives <= 3){
                    board[i][j] = 3;
                }
                if (board[i][j] == 0 && lives == 3){
                    board[i][j] = 2;
                }
            }
        }
        for (int i = 0; i <= row; i++){
            for (int j = 0; j <= col; j++){
                board[i][j] >>= 1;
            }
        }
    }
    
    private int countNeighbours (int[][] board, int row, int col, int i , int j){
        int count = 0;
        for (int x = Math.max(0, i - 1); x <= Math.min(row, i + 1); x++){
            for (int y = Math.max(0, j - 1); y <= Math.min(col, j + 1); y++){
                count += board[x][y] & 1;
            }
        }
        count -= board[i][j] & 1;
        return count;
    }
}
```

42. 70.Climbing Stairs
[LeetCode](https://leetcode.com/problems/climbing-stairs/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/78613396)
给出n阶台阶，一次可以上1阶或者2阶，计算上这n阶台阶有多少种走法。使用DP

出现面经：
[谷歌 过经](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=446929)
白人小哥，身高185+。利口起灵，比那个稍微难点，一次不是跳两级，是跳k级，给一个k，一个n，n是总台阶数，k是每次最多跳的个数，可以比它少。我一上来说dp，给了个时间复杂度 O(kn)，空间O(n)的。他说能不能简化一下空间。我说那我给了个时间kn, 空间k的。最后讨论了一阵子我简化成了时间n，空间min(n,k)的。就是不用每次都把那k个加一遍，保持一个sum就好了，每次删除一个旧的再加一个新的。

thought:
Original problem description says that we can either go 1 or 2 steps, this is a follow-up question.

code：
爬1或2是简单的dp，初始化dp 0 1， 然后dp[i] = dp[i - 1] + dp[i - 2];
**Climb 2 stairs**
```java
class Solution {
    public int climbStairs(int n) {
        if (n == 0){
            return 0;
        }
        int[] dp = new int[n + 1];
        dp[0] = dp[1] = 1;
        for (int i = 2; i <= n; i++){
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }
}
```
O(n) time and O(n) space

**Climb k stairs**
爬k阶，还是初始化，然后对于每一个i，计算从i出发，在不越界情况下，能够到达的位置，都加上dp[i]；
1. Using a dp array to contain result. dp[0] = 1; for each i before n, if it could touch to i + j where j is less equal than k and i + j won't be out of bound, dp[i + j] could count in how many ways we could get to dp[i].
```java
class Solution {
    public int climbStairs(int n, int k) {
        if (n == 0){
            return 0;
        }
        int[] dp = new int[n + 1];
        dp[0] = 1;
        for (int i = 0; i < n; i++){
            for (int j = 1; j <= k && i + j <= n; j++){
                dp[i + j] += dp[i];
            }
        }
        return dp[n];
    }
}
```
O(kn) time, O(n) space.

2.
滑动窗口的思想，维持一个大小为k的queue，其中元素是能到i的所有之前元素的方式和，初始化curSum = 1， nextSum = 0；
开始循环，从0开始，当window。size < k，queue offer curSum，然后nextSum += curSum，curSum = nextSum，这样正好是正常的曲线 1 1 2 3 5 8
当window满了，计算nextSum的时候要去掉 queue。poll（），因为我么没法从这一阶走上来了到next，也就是i+1位置了。
We could maintain a window. with size k which is current stairs which could reach to ith stairs. nextSum is the number we should offer in queue in next number, if exceed k, we should minus queue.poll() to maintain a correct value of curSum.
```java
class Solution {
    public int climbStairs(int n, int k) {
        if (n == 0){
            return 0;
        }
        Queue<Integer> window = new LinkedList<>();
        int curSum = 1;
        int nextSum = 0;
        for (int i = 0; i < n; i++){
            if (window.size() < k){
                window.offer(curSum);
                nextSum += curSum;
                curSum = nextSum;
            }
            else{
                window.offer(curSum);
                nextSum += curSum - window.poll();
                curSum = nextSum;
            }
        }
        return curSum;
    }
}
```
O(n) time, O(min(n, k)) space.


43. 392.Is Subsequence
[LeetCode](https://leetcode.com/problems/is-subsequence/description/)
[blog]()
出现面经：
[谷歌电面面经](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=461609&extra=page%3D1%26filter%3Dsortid%26sortid%3D311%26sortid%3D311)
题目：给一个品牌名和一句歌词， 按顺序使用歌词中的单词里的字母，问是否能拼出品牌名。

比如歌词是”I do not want to have to go where you do not follow”
品牌名 “wow” 就可以被拼出来 want to follow
但是品牌名 “abc”就不行

跟面试官clarify了一下，每个单词只能用一次，而且是按顺序使用，然后就讲了想法，挺简单的，面试小哥听了觉得可以，说那就不写code了，做一个更challenging的问题

其实是个follow up: 就是如果有一堆的品牌名，每一个都需要被check是不是能用这句歌词拼出来，要怎么做。讲了思路，面试官提出需要improvement, 于是调整了下思路，面试官觉得可以，然后就写了代码。

ANS:
isSubsequence变形题。
假设主串长度为N，单个品牌长度为M。一共有L个品牌。
1. 最简单的单品牌搜索，double pointer时间负责度为O(N+M).
Follow up: 多品牌
1. 多品牌一次全部给出。开一个dict记录26个字母，每个字母后面跟当前需要读入该字母的品牌。注意同个单词只能provide一次的限制。时间复杂度为O(N+L*M)。
2. 多品牌按query sequentially 地给出，需要每个字母维护其出现的index，接着用二分插找贪心判断品牌能否是子串，时间负责度为O(N + L*MlogN)。（再N远远大于M的时候效果拔群。）
这道题唯一的新意是一个单词只能provide一个字母，需要特别开条件判断。


thought:
给两个String，问s能否由t中的字符，按顺序组成。
用两个指针解决，如果indexS 和 indexT相等了才++， 能循环完s返回true，否则t循环完了就false

Using two pointers at string brand and string lyric. If they are same, index of brand move forward, and index of lyric move forward every time. If index of brand is equal to the length of brand, return true;

code：
```java
class Solution {
    public boolean isSubsequence(String s, String t) {
        if(s == null || s.length() == 0){
            return true;
        }
        int indexS = 0, indexT = 0;
        while (indexT < t.length()){
            if (s.charAt(indexS) == t.charAt(indexT)){
                indexS++;
                while(indexT < t.length() && t.charAt(indexT) != ' '){
                    indexT++;
                }
            }
            if (indexS == s.length()){
                return true;
            }
            indexT++;
        }
        return false;
    }
}
```
to solve each word could just use once, add a while loop to move indexT forward to next word.

follow-up:
给一堆s需要判断。我们把t处理一下，把t中的每个char建立一个map，记录它在t中的index，存在一个list里。
对于s的每个char，我们得到它对应的在t中的index list，prev记录了上一个char在t中的index，我们要二分查找到一个比prev大的index，如果没有就返回-1，无法组成。
```java
/**
 * Follow-up
 * If we check each sk in this way, then it would be O(kn) time where k is the number of s and t is the length of t. 
 * This is inefficient. 
 * Since there is a lot of s, it would be reasonable to preprocess t to generate something that is easy to search for if a character of s is in t. 
 * Sounds like a HashMap, which is super suitable for search for existing stuff. 
 */
public boolean isSubsequence(String s, String t) {
    if (s == null || t == null) return false;
    
    Map<Character, List<Integer>> map = new HashMap<>(); //<character, index>
    
    //preprocess t, store the index of each character.
    for (int i = 0; i < t.length(); i++) {
        char curr = t.charAt(i);
        if (!map.containsKey(curr)) {
            map.put(curr, new ArrayList<Integer>());
        }
        map.get(curr).add(i);
    }
    
    int prev = -1;  //index of previous character
    for (int i = 0; i < s.length(); i++) {
        char c = s.charAt(i);
        
        //if any character is missing in t, return false.
        if (map.get(c) == null)  {
            return false;
        } else {
            List<Integer> list = map.get(c);
            prev = binarySearch(prev, list, 0, list.size() - 1);
            if (prev == -1) {
                return false;
            }
            prev++;
        }
    }
    
    return true;
}

private int binarySearch(int index, List<Integer> list, int start, int end) {
    while (start <= end) {
        int mid = start + (end - start) / 2;
        if (list.get(mid) < index) {
            start = mid + 1;
        } else {
            end = mid - 1;
        }
    }
    
    return start == list.size() ? -1 : list.get(start);
}
```



44. 158.Read N Characters Given Read4 II - Call multiple times
[LeetCode](https://leetcode.com/problems/read-n-characters-given-read4-ii-call-multiple-times/description/)
[blog]()

[157.Read N Characters Given Read4](https://leetcode.com/problems/read-n-characters-given-read4/description/)
[read4 blog](https://blog.csdn.net/BigFatSheep/article/details/79049178)

出现面经：
[骨骼背靠背店面跪经](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=461379&extra=page%3D2)
里特扣药五扒，基本不变，只是把一次读4个变成1024个。

thought:
给一个read4 api，实现read 任意个char的功能

我们用一个全局的，size为4的buffer数组来帮助读n个字符。buffcount记录每次用read4读到字符，buffpointer用于转移buffer数据到buff

count是buff的位置指针

当buffPointer为0，读4个进来，如果读不出了，那就break，返回count

如果有数据，在count < n && buffPointer < buffCount的时候，转移数据，count++

如果这次数据能全部转移，buffPointer回0，准备继续读。

code：
```java
/* The read4 API is defined in the parent class Reader4.
      int read4(char[] buf); */

public class Solution extends Reader4 {
    /**
     * @param buf Destination buffer
     * @param n   Maximum number of characters to read
     * @return    The number of characters read
     */
    //the pointer used in buffer array, to assign value to buf.
    private int buffPoninter = 0;
    //number of chars get from read4()
    private int buffCount = 0;
    //a buffer to get data using read4()
    private char[] buffer = new char[4];
    public int read(char[] buf, int n) {
        int count = 0;
        //read n to buf
        while (count < n) {
            //if buffer pointer is 0, read new data to buffer
            if (buffPoninter == 0) {
                buffCount = read4(buffer);
            }
            //can not read anything, break.
            if (buffCount == 0) break;
            //assign corresponding data to buf.
            while (count < n && buffPoninter < buffCount) {
                buf[count++] = buffer[buffPoninter++];
            }
            //reset buffer pointer.
            if (buffPoninter >= buffCount) buffPoninter = 0;
        }
        return count;
    }
}
```

45. 947.Most Stones Removed with Same Row or Column
[LeetCode](https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/description/)
[blog]()
出现面经：
[骨骼面筋](https://www.1point3acres.com/bbs/thread-461754-1-1.html)
一道题，给你一个matrix，1的位置是有一块石头，0没有石头，现在要求是 如果这一行或者这一列有两个石头，那么我们需要删掉一个石头（1->0），知道这一行或者一列没有一个以上的石头，怎么消掉最多的石头.

例如:
0 0 0 0 0
0 1 0 0 0
0 0 1 0 1
0 1 0 0 0
0 0 0 0 1

那么答案是3

[Google MTV 全套过经](https://www.1point3acres.com/bbs/thread-461780-1-1.html?_dsign=2097985a)
3. 假设有一个棋盘(二维坐标系), 棋盘上摆放了一些石子(每个石子的坐标都为整数). 你可以remove一个石子, 当且仅当这个石子的同行或者同列还有其它石子. 输入是一个list of points.

问:
1) 给这些石子坐标, 你最多能remove多少个石子?
2) Follow-up: 若想保证remove的石子数量最大, 应按照什么顺序remove? (没有写代码)



thought:
给一个二位数组，给石子的坐标，当有石头和别的石头在同一列或者同一行的时候，我们可以remove掉它，问最多能move几块石头。

Union find 的思想，当两石头在同一列或者一行，它们就是一个联通容器，connected component，我们只需计算有多少个联通容器，最后回剩下那么多个石头， stones - # connected component就是能move的数量

用一个全局变量统计components数量，开始就是stones。length。 然后初始化一个坐标到坐标的map parent，一开始所有石头的parent都是自己。

然后对于任两块石头，用union（）连起来
循环结束，返回 stones.length - count

对于union()，用find找它们的parent，如果相同就返回，否则更新map， r2 成为 r1的父亲

对于find，如果

code：
```java
class Solution {
    private int count = 0;
    public int removeStones(int[][] stones) {
        //initialize parent map, number of component is stones.length
        Map<int[], int[]> parent = new HashMap<>();
        count = stones.length;
        //initilize parent map
        for (int[] stone : stones){
            parent.put(stone, stone);         
        }
        //union each component together.
        for (int[] s1 : stones){
            for (int[] s2 : stones){
                if (s1[0] == s2[0] || s1[1] == s2[1]){
                    union(parent, s1, s2);
                }
            }
        }
        return stones.length - count;
    }
    
    private void union(Map<int[], int[]> parent, int[] s1, int[] s2){
        //find each component name, if equal, they are in same component.
        int[] r1 = find(parent, s1), r2 = find(parent, s2);
        if (r1.equals(r2)){
            return;
        }
        //if not, union together, component number minus one
        parent.put(r1, r2);
        count--;
    }
    
    private int[] find(Map<int[], int[]> parent, int[] s){
        //in case parent union to others, we should update in each find.
        if (!parent.get(s).equals(s)){
            parent.put(s, find(parent, parent.get(s)));
        }
        return parent.get(s);
    }
}
```
Comparing each pair of stones would take O(N^2) times, and assuming your union-find takes O(logN), then the entire runtime will be O(N^2*logN)

46. 6.ZigZag Conversion
[LeetCode](https://leetcode.com/problems/zigzag-conversion/description/)
[blog](https://blog.csdn.net/BigFatSheep/article/details/79535393)
出现面经：
[狗家 11轮面试 3轮电话 + 5轮onsite + 3轮加面(onsite)](https://www.1point3acres.com/bbs/thread-438216-1-1.html)
3.medium偏向easy题  给你一个String abcdefg  和一个int row = 2 要求你按顺序打印字符
a    c     e     g            打印结果 acegbdf                          
  b     d     f
如果row = 3
a           e                打印结果  aebdfcg
  b     d      f
     c               g
虽然例子的顺序不一样，但是输出结果是一样的。


thought:
将一个正常的String，从上到下在从下到上，然后按每一行合并后输出

我们保持min（numRows， s.length）个StringBuilder，如果现在的row是0 或者 numRows - 1,就换方向，不断在对应位置的StringBuilder后面append字符，然后最后把每个StringBUilder连起来即可。
Use a list to contain each row, rearrange the order of characters.

code：
```java
class Solution {
    public String convert(String s, int numRows) {
        if (numRows == 1 || s.length <= numRows) return s;

        List<StringBuilder> rows = new ArrayList<>();
        for (int i = 0; i < numRows; i++)
            rows.add(new StringBuilder());

        int curRow = 0;
        boolean goingDown = false;

        for (char c : s.toCharArray()) {
            rows.get(curRow).append(c);
            if (curRow == 0 || curRow == numRows - 1) goingDown = !goingDown;
            curRow += goingDown ? 1 : -1;
        }

        StringBuilder res = new StringBuilder();
        for (StringBuilder row : rows) res.append(row);
        return res.toString();
    }
}
```

47. 90.Subsets II
[LeetCode](https://leetcode.com/problems/subsets-ii/description/)
[blog]()
出现面经：
[Google电面](https://www.1point3acres.com/bbs/thread-456959-1-1.html)
第二道也很简单，
subsetII, 只不过是不需要输出所有结果，返回数量就行了。
首先用recursion那种方法做的，写出一个解法。
后来一直试图搞出iterative的，但到最后也没写出来。。

[Backtracking solutions for subset/permutations](https://leetcode.com/problems/permutations/discuss/18239/A-general-approach-to-backtracking-questions-in-Java-(Subsets-Permutations-Combination-Sum-Palindrome-Partioning))


thought:
输出一个数组的所有子集。

类似于dfs的思想，res保持结果，subset是目前的子集，从0开始递归

如果pos <= nums.length，也就是可以加紧res的情况，加入
然后从pos开始，直到nums.length，在subset中加入nums[i]，调用helper，然后回退，在subset中去掉最后一个元素
为了避免重复元素，用一个while再move i forward。

recursive solution
code：
```java
class Solution {
    public List<List<Integer>> subsetsWithDup(int[] nums) {
    Arrays.sort(nums);
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> subset = new ArrayList<>();
    helper(nums, 0, res, subset);
    return res;
    }
    
    private void helper(int[] nums, int pos, List<List<Integer>> res, List<Integer> subset) {
        if (pos <= nums.length) {
            res.add(subset);
        }
        int i = pos;
        while (i < nums.length) {
            subset.add(nums[i]);
            helper(nums, i + 1, res, new ArrayList<>(subset));
            subset.remove(subset.size() - 1);
            i++;
            //avoid duplicate element
            while (i < nums.length && nums[i] == nums[i - 1]) {i++;}
        }
    }
}
```

48. 679.24 Game
[LeetCode](https://leetcode.com/problems/24-game/description/)
[blog]()
出现面经：
[Google onsite面筋](https://www.1point3acres.com/bbs/thread-461856-1-1.html)
第三轮：给一个array，里面的数字可以用任意+-*/()连接，然后问最后的结果的数量。注意顺序不可以变。
疑似变形。

[Google onsite面经](https://www.1point3acres.com/bbs/thread-462564-1-1.html)
最后一个是加拿大人小哥。给四个数字，要求相对顺序不变，可以加+-*/()，问能不能算出24。lz说可以从Polish notation角度来看，这样就不用考虑加括号了，然后递归枚举每个位置+-*/。面试官说这是个general solution，但是可能写起来要久一点，也可以就用对应的五种可能的二叉树，然后枚举三个运算符。简单写了一下框架，面试官说可以了。要注意还有一些edge case，比如中间结果出0的话，不能作为除数之类的。（lz说的时候operator operand傻傻分不清，面试官应该觉得挺好笑的hhhh）

[狗家昂赛new grad](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=450642)
第二题，是的还有第二题，大姐说没想到做挺快（啥玩意这题你原来打算讲一个小时的么…），就临时去题库里又抽了一道给我，还是LC原题但是我找不到题号了，知道的老哥可以在楼下贴一下：
给一个array of integer，可以在里面加+-*/()，问可以有多少种输出，很简单直接dfs就好了(每两个数都有可能第一个算，然后结果有四种，注意copy list)，然后印度大姐又开始了…(其实也能理解，工作这么多年了应该也不会再去碰算法题之类的东西了吧，不过…面别人之前看一眼答案还是有必要的吧…)，之后我嘴欠说可以用hashmap来improve performance（说完我就后悔了，还得给她解释，为什么我们key用String不用List）
中间可能大姐也觉得尴尬，问了我几个类似HashMap和HashTable区别这种问题…
这轮场面简直是混乱，所以我不确定feedback会怎样，也算做有悬念吧


thought:
给一个数组，表示四张牌的点数，问能否完成24点的游戏。

波兰表示法的思想，符号operator在operand之前，这样就不用考虑括号了。

先把数转成double，存进list中，然后开始dfs，

dfs，如果只剩一张牌，那就是最后结果，如果它和24.0差距 < 0.001,就是结果了，return true
如果还有多个，选两张牌，根据Poland notation，计算所有可能的结果，
对于这每一个结果，加到next round中先，然后把剩下的牌加进去，再次调用dfs(nextRound)，如果能行，就返回true。

所有可能产生的结果包括 a+b, a-b, b-a, a*b, a/b, b/a共六种。

code：
```java
class Solution {
    public boolean judgePoint24(int[] nums) {
        List<Double> list = new ArrayList<>();
        for (int i : nums) {
            list.add((double) i);
        }
        return dfs(list);
    }

    // 每次dfs都是选取两张牌
    private boolean dfs(List<Double> list) {
        if (list.size() == 1) {
            // 如果此时list只剩下了一张牌
            if (Math.abs(list.get(0)- 24.0) < 0.001) {
                return true;
            }
            return false;
        }
        
        // 选取两张牌
        for(int i = 0; i < list.size(); i++) {
            for(int j = i + 1; j < list.size(); j++) {
                // 对于每下一个可能的产生的组合
                for (double c : compute(list.get(i), list.get(j))) {
                    List<Double> nextRound = new ArrayList<>();
                    // 将组合和剩余元素加入到下一个list的dfs中循环去
                    nextRound.add(c);
                    for(int k = 0; k < list.size(); k++) {
                        if(k == j || k == i) continue;
                        nextRound.add(list.get(k));
                    }
                    if(dfs(nextRound)){
                        return true;
                    }
                }
            }
        }
        return false;

    }
    // 计算下一个可能产生的组合
    private List<Double> compute(double a, double b) {
        List<Double> res = Arrays.asList(a + b, a - b, b - a, a * b, a / b, b / a);
        return res;
    }
}
```

49. 720.Longest Word in Dictionary
[LeetCode](https://leetcode.com/problems/longest-word-in-dictionary/description/)
[blog]()
出现面经：
[Google 2019 暑期实习二面+timeline](https://www.1point3acres.com/bbs/thread-462564-1-1.html)
1. 给定一列单词， 写一个function来看一个string是不是合规的collapsible单词。 如果这个string可以每次删除一个字母（直到只有一个字母为止）都能得到一个在单词列表里的单词，那就是collapsible word。
假设单词列表里有 [great, eat, geat, a, at]
那么单词great就可以: great -> geat -> eat -> at -> a 

[狗家昂赛new grad](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=450642)
第一轮：很nice的白人小哥，简单的双方自我介绍之后开始做题：
题目给一个dictionary of words，和一个word，问你这个word可不可以每次删除一个letter直到最后只剩一个letter，每次删除的要求是删除后这个new word也在dict里面
dfs解之，improve的话注意给一个HashSet做memory就好了，剩的时间长讨论复杂度，问得比较细，这里注意hashmap.contains对String使用的时候不是O(1)
闲聊：我说拒了阿里的offer，小哥说我亏了，可以拿着那个offer从hr那儿多要些钱的（说的好像你们决定给我offer了一样……）
这一轮有信心拿个positive的


thought:
给一个String数组，找到数组中最长的，能够由字典内部的元素每次增加一个字母形成的最长的单词。
面经是题目的反过来的形式，本来是找能够由dict内word组成的最长的单词，现在变为反过来去寻找这个单词能否一步步减到只有一个字符。

首先字典排序，用一个hashset记录built的String，开始循环
如果只是一个字符，或者built里有这个word去掉最后一个字符的String，把它加入built，然后看比res长么，更新res。

code：
```java
class Solution {
    public String longestWord(String[] words) {
        //sort first
        Arrays.sort(words);
        Set<String> built = new HashSet<>();
        String res = "";
        for (String word : words){
            //if it is a single word, or built contains a previous word.
            if (word.length() == 1 || built.contains(word.substring(0, word.length() - 1))){
                res = word.length() > res.length() ? word : res;
                built.add(word);
            }
        }
        return res;
    }
}
```

50. 278.First Bad Version
[LeetCode](https://leetcode.com/problems/first-bad-version/description/)
[blog]()
出现面经：
[狗家电面面经](https://www.1point3acres.com/bbs/thread-462157-1-1.html)
大哥先问我懂不懂github, 我说懂。然后他说好，我们现在有100个commit, 当时楼主脑海闪过一道题，结果果然是：

蠡口二期吧
题目本身没什么，上来直接说二分然后说了复杂度。主要讲讲follow up吧：

“假设 test 一个 version 的操作需要很长时间，比如说 20 分钟，怎么做能减少找到 target 的时间？”

其实算是比较典型的follow up的问题，但是当时楼主没有很快想到，当面试官提示说我们有很多resource的时候才想到可以用多台机器同时跑。
这样 100 个 commit, 有 10 台机器的话就变成 10 台机器各自负责一个 range。确定一个range以后再10等分这个range。

这题没有要求code, 毕竟太简单了, 主要就是聊聊想法。


thought:
二分搜索，主要是mid是bad了不能确定，相当于 我知知道 mid >= target，我还得继续，知道low high相遇。

code：
```java
/* The isBadVersion API is defined in the parent class VersionControl.
      boolean isBadVersion(int version); */

public class Solution extends VersionControl {
    public int firstBadVersion(int n) {
        int low = 1, high = n;
        while (low < high){
            int mid = low + (high - low) / 2;
            if (isBadVersion(mid)){
                high = mid;
            }
            else{
                low = mid + 1;
            }
        }
        return low;
    }
}
```

51. 222.Count Complete Tree Nodes
[LeetCode](https://leetcode.com/problems/count-complete-tree-nodes/description/)
[blog]()
出现面经：
[狗家电面面经](https://www.1point3acres.com/bbs/thread-462157-1-1.html)
面试官说好接下来我想看一些code, 你知道binary tree吗, 知道什么是complete binary tree吗？
我说知道，然后跟他确认了下 complete binary tree 的定义。同时脑海中闪过两道题，果然是其中一道.....

蠡口尔耳儿
这道题刷过，上来也不演啥，确认了 TreeNode 的结构以后直接说了 logN * logN 的那个解法。

写完以后问复杂度，我说了，然后面试官问了一句大概是："..and it can be reduced to...?" 
这个让我有点懵逼，我的理解是想让我化简 logN * logN? 楼主表示不知道怎么化简，就说 log^2(N)，就是(logN)的平方...面试官说 ok that's fine...
然后跑了几个test cases 就开始聊天了。

thought:
给一棵完全二叉树，计算node的数量。
简单的方法可以bfs，看找到了几个node

我们的方法是，分别计算左右子树的深度。如果左右都是完全的，那么node就是 2^depth次方 - 1；如果不是，证明右边是不完全的，递归左右两颗子树加上1（root）。


code：
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
//BFS is to slow.
class Solution {
    public int countNodes(TreeNode root) {
        int leftDepth = leftDepth(root);
        int rightDepth = rightDepth(root);
        
        if (leftDepth == rightDepth){
            return (1 << leftDepth) - 1;//2^(leftDepth) - 1, it means that it is complete
        }
        else{
            return 1 + countNodes(root.left) + countNodes(root.right);
        }
    }
    
    private int leftDepth(TreeNode root){
        int depth = 0;
        while (root != null){
            root = root.left;
            depth++;
        }
        return depth;
    }
    
    private int rightDepth(TreeNode root){
        int depth = 0;
        while (root != null){
            root = root.right;
            depth++;
        }
        return depth;
    }
}
```

52. 165.Compare Version Numbers
[LeetCode](https://leetcode.com/problems/compare-version-numbers/description/)
[blog]()
出现面经：
[Google onsite](https://www.1point3acres.com/bbs/thread-461305-1-1.htmll)
2.check 版本号 输入是两个string num followed by dot followed by int 看哪个更新。
follow up：如何validate string

thought:
比较两个version number哪个更新

首先用split.("\\.")分割成string数组，然后得到最长的长度作为循环边界，如果合法呢就转成数字，否则是零，然后用compare，小的花就是-1，不等于，也就是0就返回。循环可以结束就返回0；

Simply split two version strings and parse each number to compare which is bigger.

code：
```java
class Solution {
    public int compareVersion(String version1, String version2) {
        String[] v1 = version1.split("\\.");
        String[] v2 = version2.split("\\.");
        int length = Math.max(v1.length, v2.length);
        for (int i = 0; i < length; i++){
            Integer n1 = i < v1.length ? Integer.parseInt(v1[i]) : 0;//Integer注意的，要不然没法用compareTo，还得自己写if
            Integer n2 = i < v2.length ? Integer.parseInt(v2[i]) : 0;
            int compare = n1.compareTo(n2);
            if (compare != 0){
                return compare;
            }
        }
        return 0;
    }
}
```

53. 100.Same Tree
[LeetCode](https://leetcode.com/problems/same-tree/description/)
[blog]()
出现面经：
[狗家新鲜面经](https://www.1point3acres.com/bbs/thread-462576-1-1.html)
第一道壹佰，用recursion很快写出来了
隔壁28题类似，是判断对称的。


thought:
先判断结构，2个if；再判断value，1个if；最后左右子树都满足。

code：
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null){
            return true;
        }
        if (p == null || q == null){
            return false;
        }
        if (p.val != q.val){
            return false;
        }
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }
}
```

54. 138.Copy List with Random Pointer
[LeetCode](https://leetcode.com/problems/copy-list-with-random-pointer/description/)
[blog]()
出现面经：
[狗家新鲜面经](https://www.1point3acres.com/bbs/thread-462576-1-1.html)
第二道易三巴，没想到这道题挂了，用英文没讲清楚，面试官说用中文讲吧；好不容易讲明白了，没想到刷题时候被ac的方法有问题；
之前没有仔细看discussion，时间已经到了，只能说刷题还是不到位。

thought:
用一个map，value是复制的node，第一次遍历先复制点，然后第二次遍历，复制pointer关系，注意pointer指向的都是map。value，否则就不是复制了。
Use a map to contain duplicated nodes, then use two loop to copy nodes and value first, then copy random and next pointers.

code：
```java
/**
 * Definition for singly-linked list with a random pointer.
 * class RandomListNode {
 *     int label;
 *     RandomListNode next, random;
 *     RandomListNode(int x) { this.label = x; }
 * };
 */
public class Solution {
    public RandomListNode copyRandomList(RandomListNode head) {
        if (head == null){
            return null;
        }
        Map<RandomListNode, RandomListNode> map = new HashMap<>();
        //first loop, copy all nodes value;
        RandomListNode node = head;
        while(node != null){
            map.put(node, new RandomListNode(node.label));
            node = node.next;
        }
        //second loop, copy next pointers and random pointers.
        //remember always use map to get duplicate nodes, not link to original
        node = head;
        while(node != null){
            map.get(node).next = map.get(node.next);
            map.get(node).random = map.get(node.random);
            node = node.next;
        }
        return map.get(head);
    }
}
```
[O(1) space solution](https://leetcode.com/problems/copy-list-with-random-pointer/solution/)
大概是复制链表放在next位置，然后再重新设置指针。


55. 23.Merge k Sorted Lists
[LeetCode](https://leetcode.com/problems/merge-k-sorted-lists/description/)
[blog]()
出现面经：
[狗家技术电面-十二月第一波](https://www.1point3acres.com/bbs/thread-462577-1-1.html)
第二题
要求和第一题一样，不过是给k个list， 然后merge
利口 耳弎

thought:
用一个priority queue，每次都把最小的值的链表放在队头，remove之后把剩下链表的部分放回pq中。

code：
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists.length == 0){
            return null;
        }
        PriorityQueue<ListNode> pq = new PriorityQueue<ListNode>(lists.length, new Comparator<ListNode>(){
            @Override
            public int compare(ListNode l1, ListNode l2){
                if (l1.val < l2.val){
                    return -1;
                }
                else if (l1.val == l2.val){
                    return 0;
                }
                else{
                    return 1;
                }
            }
        });
        
        ListNode dummy = new ListNode(0);
        ListNode tail = dummy;
        for (ListNode node : lists){
            if (node != null){
                pq.add(node);
            }
        }
        
        while(!pq.isEmpty()){
            tail.next = pq.poll();
            tail = tail.next;
            
            if(tail.next != null){
                pq.add(tail.next);
            }
        }
        
        return dummy.next;
    }
}
```

56. 723.Candy Crush
[LeetCode](https://leetcode.com/problems/candy-crush/description/)
[blog]()

出现面经：
[狗家上门](https://www.1point3acres.com/bbs/thread-461424-1-1.html)
round 1: 判断candy crush的initial board是否合法
第一轮上来脑子抽风把判断条件搞得和数独差不多，想了半天也不能避免backtracking，写完了最后才发现不对，最后改对了。。

[热乎乎的狗家电面](https://www.1point3acres.com/bbs/thread-461424-1-1.html)
如何初始化Candy Crush的棋盘? 限制有：
1. 开始是八乘八的棋牌；
2. 有A,B,C,D四种糖；
3. 不能有三种连续的糖在一起（上下左右相连的连续三个不行）

讲了下思路，然后小哥觉得可以就开始写代码，期间小哥给了一些提示。写完了以后开始follow up

followup 1： 如果棋盘大小和糖的种类是变量怎么办？
followup 2： 若何判断这个新初始化的棋盘的难易程度

thought:
判断candy crush 的棋盘是不是合法的，如果不是就消掉非法的，然后drop剩下的。

用一个set保存需要消除的点，然后遍历board，有6种要消除的情况，满足就加进set

如果set空，那么合法，返回board，否则全部置0然后drop，继续递归

drop的方式时col by col的，对于每一列，用两个指针，bottom表示非零元素应该放置的位置，top指针作为快指针遍历，非0就assign到bottom位置；当top到顶，再自bottom向上，全部设成0；

code：
```java
class Solution {
    public int[][] candyCrush(int[][] board) {
        Set<int[]> set = new HashSet<>();
        for (int i = 0; i < board.length; i++){
            for (int j = 0; j < board[i].length; j++){
                int cur = board[i][j];
                if (cur == 0){
                    continue;
                }
                //six circumstances that it could be crushed.
                //i j 是竖3最下面那个
                if ((i - 2 >= 0 && board[i - 1][j] == cur && board[i - 2][j] == cur) 
                //i j 是竖3最上面那个
                || (i + 2 < board.length && board[i + 1][j] == cur && board[i + 2][j] == cur) 
                //i j 是横3最右面那个
                || (j - 2 >= 0 && board[i][j - 1] == cur && board[i][j - 2] == cur) 
                //i j 是横3最左面那个
                || (j + 2 < board[i].length && board[i][j + 1] == cur && board[i][j + 2] == cur) 
                //i j 是竖3中间那个
                || (i - 1 >= 0 && i + 1 < board.length && board[i - 1][j] == cur && board[i + 1][j] == cur)
                //i j 是横3中间那个
                || (j - 1 >= 0 && j + 1 < board[i].length && board[i][j - 1] == cur && board[i][j + 1] == cur)){
                    set.add(new int[]{i, j});
                } 
            }
        }
        
        if (set.isEmpty()){
            return board;
        }
        for (int[] pos : set){
            int x = pos[0];
            int y = pos[1];
            board[x][y] = 0;
        }
        drop(board);
        return candyCrush(board);
    }
    
    private void drop(int[][] board){
        //drop cells col by col
        for (int j = 0; j < board[0].length; j++){
            int bottom = board.length - 1;
            int top = board.length - 1;
            //use two pointers to move non-empty cell down
            while (top >= 0){
                if (board[top][j] == 0){
                    top--;
                }
                else{
                    board[bottom--][j] = board[top--][j];
                }
            }
            //assign rest top value to 0.
            while (bottom >= 0){
                board[bottom--][j] = 0;
            }
        }
    }
}
```

57. 736.Parse Lisp Expression
[LeetCode](https://leetcode.com/problems/parse-lisp-expression/description/)
[blog]()
出现面经：
[狗家上门](https://www.1point3acres.com/bbs/thread-461424-1-1.html)

thought:
内们跳槽的题是真的难，不会

code：
```java
class Solution {
    public int evaluate(String expression) {
        return eval(expression, new HashMap<>());
    }
        
    private int eval(String exp, Map<String, Integer> parent) {
        if (exp.charAt(0) != '(') {
            // just a number or a symbol
            if (Character.isDigit(exp.charAt(0)) || exp.charAt(0) == '-')
                return Integer.parseInt(exp);
            return parent.get(exp);
        }
        // create a new scope, add add all the previous values to it
        Map<String, Integer> map = new HashMap<>();
        map.putAll(parent);
        List<String> tokens = parse(exp.substring(exp.charAt(1) == 'm' ? 6 : 5, exp.length() - 1));
        if (exp.startsWith("(a")) { // add
            return eval(tokens.get(0), map) + eval(tokens.get(1), map);
        } 
        else if (exp.startsWith("(m")) { // mult
            return eval(tokens.get(0), map) * eval(tokens.get(1), map);
        } 
        else { // let
            for (int i = 0; i < tokens.size() - 2; i += 2)
                map.put(tokens.get(i), eval(tokens.get(i + 1), map));
            return eval(tokens.get(tokens.size() - 1), map);
        }
    }
        
    private List<String> parse(String str) {
        // seperate the values between two parentheses
        List<String> res = new ArrayList<>();
        int par = 0;
        StringBuilder sb = new StringBuilder();
        for (char c: str.toCharArray()) {
            if (c == '(') par++;
            if (c == ')') par--;
            if (par == 0 && c == ' ') {
                res.add(new String(sb));
                sb = new StringBuilder();
            } 
            else {
                sb.append(c);
            }
        }
        if (sb.length() > 0) res.add(new String(sb));
        return res;
    }
}
```

58. 163.Missing Ranges
[LeetCode](https://leetcode.com/problems/missing-ranges/description/)
[blog]()
出现面经：
磊哥经验

thought:
给一数组，和范围lower和upper，找到在这个范围内，数组中缺失的元素，形如“a->b”的形式返回。
cornercase，不存在， 那就时loewer到upper，返回回去。
lower太大，返回回去

int next是我们要找的下一个数，next = lower first
然后循环nums，如果没到，小于next，continue；如果等于next，next++，表示这是下一个要找的；如果大于next了，不得行我们就得加进去了

如果大于等于upper了，表示已经到了，可以直接推出；next = nums[i] + 1，表示下一个数要找的。

推出循环之后判断，如果next还没到upper，也就是upper比所有数都大，再加一个

关于那个添加结果，如果一样就返回String的这个数，不一样就连起来返回。
next is the number we need to find, if we can't find it, we add [next, nums[i] - 1] to res list.

code：
```java
class Solution {
    public List<String> findMissingRanges(int[] nums, int lower, int upper) {
        List<String> res = new ArrayList<>();
        
        //corner case, nums is empty and lower is out of bound, no result.
        if (nums.length == 0){
            res.add(getRange(lower, upper));
            return res;
        }
        if (lower == Integer.MAX_VALUE){
            return res;
        }
        //the next number we need to find;
        int next = lower;
        for (int i = 0; i < nums.length; i++){
            //have not reach the range.
            if (nums[i] < next){
                continue;
            }
            if (nums[i] == next){
                next++;
                continue;
            }
            
            //get the missing range String format
            res.add(getRange(next, nums[i] - 1));
            
            //Overflow, just return
            if (nums[i] >= upper){
                return res;
            }
            
            //find next number
            next = nums[i] + 1;
        }
        
        //check if upper bigger than the biggest element in the nums array
        if (next <= upper){
            res.add(getRange(next, upper));
        }
        return res;
    }
    
    private String getRange(int n1, int n2){
        return n1 == n2 ? String.valueOf(n1) : String.valueOf(n1) + "->" + String.valueOf(n2);
    }
}
```

59. path sum 系列
[LeetCode]()
[blog]()

出现面经：
[狗家11月7号on-site 已签](https://www.1point3acres.com/bbs/thread-462598-1-1.html)
小哥看了一下代码 说我写的太快了 再出一题吧
binary tree path sum
input binary tree node， int sum
output boolean true（如果有path 的和等于 sum ） 反之 return false
lc上有原题  就是一个 遍历 
又秒掉 。还剩了20分钟 小哥说我也没准备那么多题 我们聊聊天吧 就问了问楼主现在做的东西之类的 耗一下时间。。。。

[112. Path Sum](https://leetcode.com/problems/path-sum/description/)
判断是否有root to leaf path sum 为 target
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null){
            return false;
        }
        if (root.left == null && root.right == null && root.val == sum){
            return true;
        }
        return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
    }
}
```

[113. Path Sum II](https://leetcode.com/problems/path-sum-ii/description/)
输出所有root-to-leaf paths ，sum 为 sum的路径。
DFS 这棵树，如果到底满足就入res，一定记得remove path，为了回退，因为我们只有一个path。
加入res的时候要new 一个新的，否则就不对了。
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    List<List<Integer>> res;
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        res = new ArrayList<>();
        dfsHelper(root, new ArrayList<>(), 0, sum);
        return res;
    }
    
    private void dfsHelper(TreeNode node, List<Integer> path, int curSum, int sum){
        if (node == null){
            return;
        }
        curSum += node.val;
        path.add(node.val);
        if (node.left == null && node.right == null && curSum == sum){
            res.add(new ArrayList<>(path));
        }
        else {
            dfsHelper(node.left, path, curSum, sum);
            dfsHelper(node.right, path, curSum, sum);
        }
        path.remove(path.size() - 1);
    }
}
```

[437. Path Sum III](https://leetcode.com/problems/path-sum-iii/description/)
path不一定由root开始，只要能凑成sum就行，而且里面有负数。
用一个preSum的map来存储之前出现的path sum值出现的次数，用（0， 1）来初始化，保证碰到合适的能加到res上。然后递归左右子树得到结果，重新更新preSum保证回退的结果。
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public int pathSum(TreeNode root, int sum) {
        Map<Integer, Integer> preSum = new HashMap<>();
        preSum.put(0, 1);
        return helper(root, 0, sum, preSum);
    }
    
    private int helper(TreeNode root, int curSum, int target, Map<Integer, Integer> preSum){
        if (root == null){
            return 0;
        }
        
        curSum += root.val;
        //找之前出现过的curSum - target出现的 次数，本源就是（0，1）
        int res = preSum.getOrDefault(curSum - target, 0);
        preSum.put(curSum, preSum.getOrDefault(curSum, 0) + 1);
        
        res += helper(root.left, curSum, target, preSum) + helper(root.right, curSum, target, preSum);
        preSum.put(curSum, preSum.get(curSum) - 1);
        return res;
    }
}
```



60. 299.Bulls and Cows
[LeetCode](https://leetcode.com/problems/bulls-and-cows/description/)
[blog]()
出现面经：
[狗家11月7号on-site 已签](https://www.1point3acres.com/bbs/thread-462598-1-1.html)
压抑小胖姐姐 （很man 的一个小胖姐姐 ）
废话不多  先问了一个bulls and cows 刷题网的原题 
秒掉。。。。
然后 问有什么好的strategy 可以尽快的猜出这个词。。。。楼主懵逼
说了几个解决方法 一个一个字母试 复杂度 n*256 或者 先猜出来这个词的anagram 然后排序啊
反正小姐姐怎么都说不行 能不能再优化。。。。 
楼主继续拉格朗日懵逼完全不知道干嘛。。。。。
尴尬中结束。。。
然后最后还是神奇的过了。。。

开始bulls and cows 是两个string 
follow up 就给一个api call guess（String word） 尽可能减少call的次数 猜出secret string


thought:
位置和数字都对了是bulls，数字对了位置不对是cows，问有几个cows，几个bulls
字符相同，bulls
然后用一个array存之前的情况，如果s有，就numbers[s]++；如果g有，就numbers[g]--；然后判断，小于0了，s有说明之前g有，cows++；反之亦然。

code：
```java
class Solution {
    public String getHint(String secret, String guess) {
        int bulls = 0;
        int cows = 0;
        int[] numbers = new int[10];
        for (int i = 0; i < secret.length(); i++){
            int s = secret.charAt(i) - '0';
            int g = guess.charAt(i) - '0';
            if (s == g){
                bulls++;
            }
            else{
                //if character is not match
                //if secret have this char and it original value is less than 0, it means guess have this char before, cows++;
                if (numbers[s]++ < 0){
                    cows++;
                }
                //same, if secret have this char before, cows++, and numbers[g] should --, they are vice.
                if (numbers[g]-- > 0){
                    cows++;
                }
            }
        }
        return bulls + "A" + cows + "B";
    }
}
```

61. 528.Random Pick with Weight 
[LeetCode](https://leetcode.com/problems/random-pick-with-weight/description/)
[blog]()
出现面经：
[骨骼新鲜电面](https://www.1point3acres.com/bbs/thread-462699-1-1.html?_dsign=5dc9a1c0)
Weighted Random. 鲤窛污迩拔

Constructor Input: Object[] objects, int[] weights
Goal: Implement a getWeightedRandom() method that randomly returns an object with the corresponding probability (P(i) = weights / sum(weights)) 

Preprocessing 和 getWeightedRandom()的复杂度 各种情况从简单到复杂都过一遍。最优解见蠡口。

thought:
在有权重的情况下，随机取数字。

给w个正数，w[i]为i的权重，按照权重随机取一个数。计算所有的权重和，然后进行binary search，当value在sum[i - 1] 和 sum[i],之间时，因为sum[i]是包含了w[i]的权重，所以left = mid + 1， 返回left值。如果正好相等，返回mid。可以画一个叠起来的柱状图帮助理解。

code：
```java
class Solution {
    int[] sum;
    Random rand;
    public Solution(int[] w) {
        this.rand = new Random();
        for (int i = 1; i < w.length; i++){
            w[i] += w[i - 1];
        }
        this.sum = w;
    }
    
    public int pickIndex() {
        //value is a integer between [1, maxSum]
        int value = rand.nextInt(sum[sum.length - 1]) + 1;
        int left = 0, right = sum.length - 1;
        while (left < right){
            int mid = (left + right) / 2;
            if (sum[mid] == value){
                return mid;
            }
            else if (sum[mid] > value){
                right = mid;
            }
            else{
                left = mid + 1;
            }
        }
        return left;
    }
}

/**
 * Your Solution object will be instantiated and called as such:
 * Solution obj = new Solution(w);
 * int param_1 = obj.pickIndex();
 */
```

62. 221.Maximal Square
[LeetCode](https://leetcode.com/problems/maximal-square/description/)
[blog]()
出现面经：
[咕咕电面](https://www.1point3acres.com/bbs/thread-461070-1-1.html?_dsign=12ed2e43)
第二题

给一个char grid，里面 'b'表示黑色，'w'表示白色，返回黑色形成的最大的正方形的右下角坐标。
followup， 如果要找的是最大的矩形怎么办

thought:
返回二位数组中由1组成的最大的正方形的面积

dp解决，dp[i][j]表示 位置i， j处作为正方形的右下角，能给形成的最大正方形的边长，显然，它的左，上，左上的最小值，是这个正方形的边长，因为这样相当于在正方形下方和右方又包裹上一层。

计算dp，然后记录最大值。

code：
```java
class Solution {
    public int maximalSquare(char[][] matrix) {
        if (matrix.length == 0){
            return 0;
        }
        int res = 0, m = matrix.length, n = matrix[0].length;
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++){
            for (int j = 1; j <= n; j++){
                //index starts at 0，so minus one
                if (matrix[i - 1][j - 1] == '1'){
                    dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1])) + 1;
                }
                res = Math.max(res, dp[i][j]);
            }
        }
        return res * res;
    }
}
```

follow-up
[85. Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/description/)
二维数组中由1组成的最大长方形面积???????????????
记录每一列的左侧index，右侧index，heigth，计算选出在这一行能形成的最大的长方形。

```java
class Solution {
    public int maximalRectangle(char[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0){
            return 0;
        }
        int m = matrix.length, n = matrix[0].length;
        int[] left = new int[n], right = new int[n], height = new int[n];
        Arrays.fill(right, n);
        int res = 0;
        for (int i = 0; i < m; i++){
            int curLeft = 0, curRight = n;
            for (int j = 0; j < n; j++){
                if (matrix[i][j] == '1'){
                    height[j]++;                    
                    left[j] = Math.max(left[j], curLeft);
                }
                else{
                    height[j] = 0;
                    
                    left[j] = 0;
                    curLeft = j + 1;
                }
                int k = n - 1 - j;
                if (matrix[i][k] == '1'){
                    right[k] = Math.min(right[k], curRight);
                }
                else{
                    right[k] = n;
                    curRight = k;
                }
            }

            for (int j = 0; j < n; j++){
                res = Math.max(res, (right[j] - left[j]) * height[j]);
            }
        }
        return res;
    }
}
```


63. 369.Plus One Linked List
[LeetCode](https://leetcode.com/problems/plus-one-linked-list/description/)
[blog]()
出现面经：
[狗狗11/27新鲜电面](https://www.1point3acres.com/bbs/thread-462455-1-1.html?_dsign=4e532d02)
linkedlist plus one, lc 散留久
follow up
两个等长linkedlist 相加, 参考lc思思无, 递归搞定

thought:
实现listnode表示的数+1 的操作，设置一个dummy 头，找到最后一个不为9的数，将它++，剩下的9置零。

code：
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode plusOne(ListNode head) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode i = dummy;
        ListNode j = dummy;
        //find last non-9 node, rest 9 would be assign to 0
        while (j != null){
            if (j.val < 9){
                i = j;
            }
            j = j.next;
        }
        i.val++;
        j = i.next;
        while (j != null){
            j.val = 0;
            j = j.next;
        }
        return dummy.val == 0 ? dummy.next : dummy;
    }
}
```

64. 445.Add Two Numbers II 
[LeetCode](https://leetcode.com/problems/add-two-numbers-ii/description/)
[blog]()
出现面经：
[狗狗11/27新鲜电面](https://www.1point3acres.com/bbs/thread-462455-1-1.html?_dsign=4e532d02)
follow up
两个等长linkedlist 相加, 参考lc思思无, 递归搞定

thought:
将两个链表表示的数相加，依然用链表表示结果。

用两个栈来存储val，先把两个数都入栈，然后不断出栈计算，加上新的node。

code：
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        Stack<Integer> s1 = new Stack<>();
        Stack<Integer> s2 = new Stack<>();
        while (l1 != null){
            s1.push(l1.val);
            l1 = l1.next;
        }
        while (l2 != null){
            s2.push(l2.val);
            l2 = l2.next;
        }
        int sum = 0;
        ListNode list = new ListNode(0);
        while (!s1.empty() || !s2.empty()){
            if (!s1.empty()){
                sum += s1.pop();
            }
            if (!s2.empty()){
                sum += s2.pop();
            }
            //assign list，add head, move list to head, reduce sum to carry
            list.val = sum % 10;
            ListNode head = new ListNode(sum / 10);
            head.next = list;
            list = head;
            sum /= 10;
        }
        return list.val == 0 ? list.next : list;
    }
}
```

65. 777.Swap Adjacent in LR String
[LeetCode](https://leetcode.com/problems/swap-adjacent-in-lr-string/description/)
[blog]()
出现面经：
[Google MTV 全套过经](https://www.1point3acres.com/bbs/thread-461780-1-1.html?_dsign=2097985a)


thought:  
XL可以转成LX，RX可以转为XR，问能否转换成end。
因为L R没法呼唤，所以它们的相对位置是确定的，如果说去掉X之后不一样，那么肯定不能转

然后用两个指针分别遍历start， end，当是X的时候疯狂前进，如果都到底了，返回true，只有一个到了返回false

如果找到的在start和end的char不一样，false；如果都找到的是L，因为L只能往左走，所以p2 > p1更大了就false；
同理如果是R，只能往后，p1 > p2了也是false。

两指针move on，进入新的征程。
code：
```java
class Solution {
    public boolean canTransform(String start, String end) {
        if (!start.replace("X", "").equals(end.replace("X", ""))){
            return false;
        }
        int p1 = 0;
        int p2 = 0;
        
        while (p1 < start.length() && p2 < end.length()){
            // get the non-X positions of 2 strings
            while(p1 < start.length() && start.charAt(p1) == 'X'){
                p1++;
            }
            while(p2 < end.length() && end.charAt(p2) == 'X'){
                p2++;
            }
            
            //if both of the pointers reach the end the strings are transformable
            if(p1 == start.length() && p2 == end.length()){
                return true;
            }
            // if only one of the pointer reach the end they are not transformable
            if(p1 == start.length() || p2 == end.length()){
                return false;
            }
            
            if(start.charAt(p1) != end.charAt(p2)){
                return false;
            }
            // if the character is 'L', it can only be moved to the left. p1 should be greater or equal to p2.
            if(start.charAt(p1) == 'L' && p2 > p1){
                return false;
            }
            // if the character is 'R', it can only be moved to the right. p2 should be greater or equal to p1.
            if(start.charAt(p1) == 'R' && p1 > p2){
                return false;
            }
            p1++;
            p2++;
        }
        return true;
    }
}
```

66. 15.3Sum
[LeetCode](https://leetcode.com/problems/3sum/description/)
[blog]()
出现面经：
程哥面经
3 sum less than k


thought:
先把数组排序，然后两个循环，第一个循环固定一个元素，然后用两个指针去找剩下的结果，找到了不要停，可能还有要继续，注意要确定第一个元素时要排除重复情况。找到3个元素的时候也要去除重复情况。

code：
```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < nums.length - 2; i++){
            // we do not want to choose repeat element
            if (i == 0 || (i > 0 && nums[i] != nums[i - 1])){
                int low = i + 1, high = nums.length - 1;
                int sum = 0 - nums[i];
                while (low < high){
                    //do not stop searching, casuse we may have other solution set
                    if (nums[low] + nums[high] == sum){
                        res.add(Arrays.asList(nums[i], nums[low], nums[high]));
                        while (low < high && nums[low] == nums[low + 1]){
                            low++;
                        }
                        while (low < high && nums[high] == nums[high - 1]){
                            high--;
                        }
                        low++;
                        high--;
                    }
                    else if (nums[low] + nums[high] < sum){
                        low++;
                    }
                    else{
                        high--;
                    }
                }
            }
        }
        return res;
    }
}
```

67. 279.Perfect Suqares
[LeetCode](https://leetcode.com/problems/perfect-squares/description/)
[blog]()
Example 1:

Input: n = 12
Output: 3 
Explanation: 12 = 4 + 4 + 4.
Example 2:

Input: n = 13
Output: 2
Explanation: 13 = 4 + 9.
出现面经：
[狗家面经](https://www.1point3acres.com/bbs/thread-429694-1-1.html?_dsign=518c1cf9)
然后又出了一个完美平方数那个题，dp解了，但是没有时间了，只能草草写完。回来之后发现初始化的时候其实有bug。哎。。哭

thought:
问一个数最少可以由几个平方数组成，使用dp解决，dp表示这个i可以由几个平方数，一开始都设成最大值。对于每一个位置的i，计算所有小于它的平方数加和用的数最少的结果。

code：
```java
class Solution {
    public int numSquares(int n) {
        if (n <= 0){
            return 0;
        }
        
        int[] dp = new int[n + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for (int i = 0; i <= n; i++){
            for (int j = 1; j * j <= i; j++){
                dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
            }
        }
        return dp[n];
    }
}
```

68. 361.Bomb Enemy
[LeetCode](https://leetcode.com/problems/bomb-enemy/description/)
[blog]()
出现面经：
[骨骼面经](https://www.1point3acres.com/bbs/thread-364274-1-1.html?_dsign=344dd110)
一：鲜花和雕像
在2D的矩阵上，某些坐标有鲜花（F），某些坐标又雕像(S)。其余的坐标是空白的（O）
例：
OOFOO
OOSOO
FFOFF
SOOOF
OFOOO

雕像会挡住这个方向的鲜花, 问：在每一个空白位置上，朝四个方向看，一共能看到几朵鲜花。
在这个例子里：在左下角的空白，朝上被雕像挡住，朝右右1朵鲜花，一共1朵
在正中间的空白，上0，左2，右2，下0，一共4朵
输出一个2d矩阵

follow up: 
1. 写test case，conrer case.
2. 如果矩阵很大，不能一次性读入内存怎么办。假设矩阵是稀疏的（鲜花和雕像很少）

thought:
二维数组中，有墙W，E敌人，空地0，炸弹只能扔在空地上，杀死直到墙或者边界上的同一行和列的所有敌人。问最多能杀死多少。

我们可以对于每一个空地，都向四个方向搜索，统计出现的敌人，但是会造成重复计算。

我们发现，在碰到墙之前，这一行或者一列能杀死的敌人数量都是一样的，所以我们可以在墙出现之后或一行开始之前，计算出值，等到出现空地就直接用就行了

因为我们的循环，行再前，所以row一个就够了，但是对应的每一列需要col[]来存储，

然后在双层循环中，是墙就过；如果是第一个元素，或者前面是墙，计算一波敌人数存起来；然后是空地就找最大值。

计算的方式很简单，只需要往后找，因为前面是墙，行列都是这样。
code：
```java
class Solution {
    //O(mn) time, O(n) space
    public int maxKilledEnemies(char[][] grid) {
        if (grid.length == 0 || grid[0].length == 0){
            return 0;
        }
        int max = 0;
        int row = 0;
        int[] col = new int[grid[0].length];
        for (int i = 0; i < grid.length; i++){
            for (int j = 0; j < grid[0].length; j++){
                if (grid[i][j] == 'W'){
                    continue;
                }
                if (j == 0 || grid[i][j - 1] == 'W'){
                    row = killedEnemiesRow(grid, i, j);
                }
                if (i == 0 || grid[i - 1][j] == 'W'){
                    col[j] = killedEnemiesCol(grid, i, j);
                }
                if (grid[i][j] == '0'){
                    max = Math.max(max, row + col[j]);
                }
            }
        }
        return max;
    }
    
    private int killedEnemiesRow(char[][] grid, int i, int j){
        int num = 0;
        while (j <= grid[0].length - 1 && grid[i][j] != 'W'){
            if (grid[i][j] == 'E'){
                num++;
            }
            j++;
        }
        return num;
    }
    
    private int killedEnemiesCol(char[][] grid, int i, int j){
        int num = 0;
        while (i <= grid.length - 1 && grid[i][j] != 'W'){
            if (grid[i][j] == 'E'){
                num++;
            }
            i++;
        }
        return num;
    }
}
```

69.  465.Optimal Account Balancing
[LeetCode](https://leetcode.com/problems/optimal-account-balancing/description/)
[blog]()
出现面经：
学姐面经

thought:
一组人互相借钱，用三元组表示x给了y z刀， eg: [[0, 1, 10], [2, 0, 5]]

给一组这样的互相借钱的情况，问需要几次交易能够平账

code：
```java
class Solution {
    public int minTransfers(int[][] transactions) {
        if(transactions == null || transactions.length == 0){
            return 0;
        }
        //use a map to count each person's account money
        Map<Integer, Integer> accounts = new HashMap<>();
        for (int i = 0; i < transactions.length; i++){
            int id1 = transactions[i][0];
            int id2 = transactions[i][1];
            int money = transactions[i][2];
            accounts.put(id1, accounts.getOrDefault(id1, 0) - money);
            accounts.put(id2, accounts.getOrDefault(id2, 0) + money);
        }
        //divide money of account into two list.
        List<Integer> negatives = new ArrayList<>();
        List<Integer> positives = new ArrayList<>();
        for (Integer id : account.keySet()){
            int money = account.get(id);
            if (money == 0){
                continue;
            }
            if (m < 0){
                negatives.add(-m);
            }
            else{
                positive.add(m);
            }
        }
        //use two stacks to help find min transactions
        int res = Integer.MAX_VALUE;
        Stack<Integer> stackNeg = new Stack<>();
        Stack<Integer> stackPos = new Stack<>();
        while(true){
            for (Integer num : negatives){
                stackNeg.push(num);
            }
            for (Integer num : positives){
                stackPos.push(num);
            }
            int cur = 0;
            while (!stackNeg.empty()){
                int n = stackNeg.pop();
                int p = stackPos.pop();
                cur++;
                if (n == p){
                    continue;
                }
                if (n > p){
                    stackNeg.push(n - p);
                }
                else{
                    stackPos.push(p - n);
                }
            }
            
        }
    }
    
}
```

70. 807.Max Increase to Keep City Skyline
[LeetCode](https://leetcode.com/problems/max-increase-to-keep-city-skyline/description/)
[blog]()
出现面经：
学姐面经

thought:
给一二维数组，每个元素表示这个位置的楼的高度，我们可以给任意一个楼加高，但是最后形成的skyline 还是要一样的。

用两个数组rowmax，colmax表示这一行的最高值和这一列的最高值
第一个循环，找到每一个行和列的最高值，存在两个数组中。
第二个循环，计算能加多少，每个位置能加的高度是它和 rowmax，colmax中的最小值的差值。

code：
```java
class Solution {
    public int maxIncreaseKeepingSkyline(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0){
            return 0;
        }
        int m = grid.length;
        int n = grid[0].length;
        int[] rowMax = new int[m];
        int[] colMax = new int[n];
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                rowMax[i] = Math.max(rowMax[i], grid[i][j]);
                colMax[j] = Math.max(colMax[j], grid[i][j]);
            }
        }
        int res = 0;
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                res += Math.min(rowMax[i], colMax[j]) - grid[i][j];
            }
        }
        return res;
    }
}
```

71. 317.Shortest Distance from All Buildings
[LeetCode](https://leetcode.com/problems/shortest-distance-from-all-buildings/description/)
[blog]()
出现面经：
[Google 电面 过经](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=468744&extra=page%3D1%26filter%3Dsortid%26sortid%3D311%26searchoption%5B3086%5D%5Bvalue%5D%3D9%26searchoption%5B3086%5D%5Btype%5D%3Dradio%26searchoption%5B3088%5D%5Bvalue%5D%3D1%26searchoption%5B3088%5D%5Btype%5D%3Dradio%26searchoption%5B3046%5D%5Bvalue%5D%3D1%26searchoption%5B3046%5D%5Btype%5D%3Dradio%26searchoption%5B3109%5D%5Bvalue%5D%3D1%26searchoption%5B3109%5D%5Btype%5D%3Dradio%26sortid%3D311%26orderby%3Ddateline)


thought:
给一个二维数组，0表示空地， 1表示房子，2表示障碍，要在空地中找一个离所有其他房子的距离和最大的点建新房子。

code：
```java
class Solution {
    class Tuple{
        public int x;
        public int y;
        public int dist;
        public Tuple(int x, int y, int dist){
            this.x = x;
            this.y = y;
            this.dist = dist;
        }
    }
    
    private int[][] dirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    public int shortestDistance(int[][] grid) {
        if (grid == null || grid.length ==  0 || grid[0].length == 0){
            return -1;
        }
        int m = grid.length;
        int n = grid[0].length;
        int[][] distance = new int[m][n];
        List<Tuple> buildings = new ArrayList<>();
        //add all buildings into a list
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                if (grid[i][j] == 1){
                    buildings.add(new Tuple(i, j, 0));
                }
                grid[i][j] = -grid[i][j];
            }
        }
        //bfs each building
        for (int k = 0; k < buildings.size(); k++){
            bfs(buildings.get(k), k, distance, grid, m, n);
        }
        //find minimum distance;
        int result = -1;
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                if (grid[i][j] == buildings.size() && (result < 0 || distance[i][j] < result)){
                    result = distance[i][j];
                }
            }
        }
        return result;
    }
    
    private void bfs(Tuple root, int k, int[][] distance, int[][] grid, int m, int n){
        Queue<Tuple> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()){
            Tuple cell = queue.poll();
            distance[cell.x][cell.y] += cell.dist;
            for (int[] dir : dirs){
                int x = cell.x + dir[0];
                int y = cell.y + dir[1];
                if (x >= 0 && x < m && y >= 0 && y < n && grid[x][y] == k){
                    grid[x][y] = k + 1;
                    queue.add(new Tuple(x, y, cell.dist + 1));
                }
            }
        }
    }
}
```

72. 82. Remove Duplicates from Sorted List II
[LeetCode](https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/description/)
[blog]()
出现面经：
[Google技术电面2019 Intern总结 求大米打赏](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=468194&extra=page%3D1%26filter%3Dsortid%26sortid%3D311%26searchoption%5B3086%5D%5Bvalue%5D%3D9%26searchoption%5B3086%5D%5Btype%5D%3Dradio%26searchoption%5B3088%5D%5Bvalue%5D%3D1%26searchoption%5B3088%5D%5Btype%5D%3Dradio%26searchoption%5B3046%5D%5Bvalue%5D%3D1%26searchoption%5B3046%5D%5Btype%5D%3Dradio%26searchoption%5B3109%5D%5Bvalue%5D%3D1%26searchoption%5B3109%5D%5Btype%5D%3Dradio%26sortid%3D311%26orderby%3Ddateline)
第一题感觉还不太难，给定一个linked list, 如果连续element重复出现，则将该元素全部删 ，类似leetcode 82， 好像之前有人碰到过。

thought:
给一个排序的链表，如果元素重复了，就把他们都删掉。

设置一个dummy头，然后两个指针，pre和cur，当cur和next相同时，cur前进，统计重复的次数；
如果重复次数大于1， 把pre。next连到cur。next；
否则移动pre到cur， cur到next

code：
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null){
            return head;
        }
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode pre = dummy;
        ListNode cur = dummy.next;
        while (cur != null){
            int count = 1;
            while (cur.next != null && cur.next.val == cur.val){
                cur = cur.next;
                count++;
            }
            if (count > 1){
                pre.next = cur.next;
                cur = cur.next;
            }
            else{
                pre = cur;
                cur = cur.next;
            }
        }
        return dummy.next;
    }
}
```

73. 4.Median of Two Sorted Arrays
[LeetCode]()
[blog]()
出现面经：
[Google技术电面2019 Intern总结 求大米打赏](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=468194&extra=page%3D1%26filter%3Dsortid%26sortid%3D311%26searchoption%5B3086%5D%5Bvalue%5D%3D9%26searchoption%5B3086%5D%5Btype%5D%3Dradio%26searchoption%5B3088%5D%5Bvalue%5D%3D1%26searchoption%5B3088%5D%5Btype%5D%3Dradio%26searchoption%5B3046%5D%5Bvalue%5D%3D1%26searchoption%5B3046%5D%5Btype%5D%3Dradio%26searchoption%5B3109%5D%5Bvalue%5D%3D1%26searchoption%5B3109%5D%5Btype%5D%3Dradio%26sortid%3D311%26orderby%3Ddateline)
类似Median of Two Sorted Arrays的一道题，

之前看过，但是自己没一步一步写写，在印度小哥的提醒下也是磕磕碰碰搞了好久，当时就觉得不好了。

thought:
经典难题啊，找到两个已经排序的数组中的中位数。

code：
```java
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if (nums1.length > nums2.length){
            return findMedianSortedArrays(nums2, nums1);
        }
        int l1 = nums1.length, l2 = nums2.length;
        int imin = 0, imax = nums1.length, halfSum = (l1 + l2 + 1) / 2;//右偏一下，表示j-1是中位数左边的数。
        while (imin <= imax){
            int i = (imin + imax) / 2;
            int j = halfSum - i;
            if (i < l1 && nums2[j - 1] > nums1[i]){// i is too small
                imin = i + 1;
            }
            else if (i > 0 && nums1[i - 1] > nums2[j]){//i is too big
                imax = i - 1;
            }
            else{
                int leftMax = Integer.MIN_VALUE;
                if (i == 0){
                    leftMax = nums2[j - 1];
                }
                else if (j == 0){
                    leftMax = nums1[i - 1];
                }
                else{
                    leftMax = Math.max(nums1[i - 1], nums2[j - 1]);
                }
                if ((l1 + l2) % 2 == 1){//odd sum number
                    return leftMax;
                }
                
                int rightMin = Integer.MAX_VALUE;
                if (i == l1){
                    rightMin = nums2[j];
                }
                else if (j == l2){
                    rightMin = nums1[i];
                }
                else{
                    rightMin = Math.min(nums1[i], nums2[j]);
                }
                return (rightMin + leftMax) / 2.0;
            }
        }
        return 0;
    }
}
```

74. 349.Intersection of Two Arrays
[LeetCode](https://leetcode.com/problems/intersection-of-two-arrays/description/)
[blog]()
出现面经：
[G家虾图昂赛](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=466779&extra=page%3D1%26filter%3Dsortid%26sortid%3D311%26searchoption%5B3086%5D%5Bvalue%5D%3D9%26searchoption%5B3086%5D%5Btype%5D%3Dradio%26searchoption%5B3088%5D%5Bvalue%5D%3D1%26searchoption%5B3088%5D%5Btype%5D%3Dradio%26searchoption%5B3046%5D%5Bvalue%5D%3D1%26searchoption%5B3046%5D%5Btype%5D%3Dradio%26searchoption%5B3109%5D%5Bvalue%5D%3D1%26searchoption%5B3109%5D%5Btype%5D%3Dradio%26sortid%3D311%26orderby%3Ddateline)


thought:
找到两个数组交叉的部分。

用一个set contains nums1，再把nums2的部分放到 另一个set， 然后存回数组返回。

code：
```java
class Solution {
    public int[] intersection(int[] nums1, int[] nums2) {
        Set<Integer> set = new HashSet<>();
        Set<Integer> intersect = new HashSet<>();
        for (int num : nums1){
            set.add(num);
        }
        for (int num : nums2){
            if (set.contains(num)){
                intersect.add(num);
            }
        }
        int[] arr = new int[intersect.size()];
        int i = 0;
        for (Integer num : intersect){
            arr[i++] = num;
        }
        return arr;
    }
}
```


75. 739.Daily Temperatures
[LeetCode](https://leetcode.com/problems/daily-temperatures/description/)
[blog]()
出现面经：
[狗家昂赛new grad](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=450642)
第四轮：印度大姐，我滴个龟龟，噩梦开始了
首先她这个口音还是有一些的，所以有时候她干脆把某些key words写在纸上…
然后，这大姐我咋觉得答案都没看就来面我…两道题：
第一题是LC739原题，弟中弟，我想这题实在没啥好说的，写起来肯定很快，我抻一下吧，就开始给她说，我要开始brute force了！刚说完大姐就说，不用不用，别brute force，你就说你觉得好的方法……那还说啥，直接按LC解法秒了啊，我没想到的是大姐完全不知道这个解法，讲解过程中屡次蒙蔽，甚至一直怀疑算法的正确性，while(true) {doTest();}，在尝试了多个test case都work的情况下大姐没脾气了，然后问我复杂度，为啥是O(n)，这地方花了好长时间才给她弄明白……

thought:
给一个数组表示每天的温度，计算每天要多久之后才会有更温暖的时候。

用一个stack存之前的每一天的index，while 有一天比栈顶元素大，出栈并且计算天数，然后把这一天入栈。

code：
```java
class Solution {
    public int[] dailyTemperatures(int[] T) {
        int[] res = new int[T.length];
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < T.length; i++){
            while (!stack.isEmpty() && T[i] > T[stack.peek()]){
                int index = stack.pop();
                res[index] = i - index;
            }
            stack.push(i);
        }
        return res;
    }
}
```

76. 300. Longest Increasing Subsequence
[LeetCode](https://leetcode.com/problems/longest-increasing-subsequence/description/)
[blog]()
出现面经：
[Google SETI 加面](https://www.1point3acres.com/bbs/thread-178258-1-1.html)


thought:
找到数组中最长的递增序列。

dp表示到这个数，最长的递增序列的长度。

dp解决，对于每个数，去前面找一个最大的加1

code：
```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        if (nums.length <= 1){
            return nums.length;
        }
        int[] dp = new int[nums.length];
        for (int i = 0; i < nums.length; i++){
            dp[i] = 1;
        }
        for (int i = 1; i < nums.length; i++){
            for (int j = 0; j < i; j++){
                if (nums[j] < nums[i]){
                    if (dp[j] + 1 > dp[i]){
                        dp[i] = dp[j] + 1;
                    }
                }
            }
        }
        int max = Integer.MIN_VALUE;
        for (int i : dp){
            if (i > max){
                max = i;
            }
        }
        return max;
    }
}
```

77. 171. Excel Sheet Column Number
[LeetCode](https://leetcode.com/problems/excel-sheet-column-number/description/)
[blog]()
出现面经：
[Google SETI 加面](https://www.1point3acres.com/bbs/thread-178258-1-1.html)



thought:
把excel的列号转为数字。

code：
```java
class Solution {
    public int titleToNumber(String s) {
        int res = 0;
        if (s.length() == 0){
            return res;
        }
        for (char ch : s.toCharArray()){
            res = res * 26 + (ch - 'A' + 1);
        }
        return res;
    }
}
```

78. 409. Longest Palindrome
[LeetCode](https://leetcode.com/problems/longest-palindrome/description/)
[blog]()
出现面经：
[Google SETI 加面](https://www.1point3acres.com/bbs/thread-178258-1-1.html)



thought:

code：
```java
class Solution {
    public int longestPalindrome(String s) {
        Set<Character> set = new HashSet<>();
        int count = 0;
        for (char ch : s.toCharArray()){
            if (set.contains(ch)){
                set.remove(ch);
                count++;
            }
            else{
                set.add(ch);
            }
        }
        return set.size() == 0 ? 2 * count : 2 * count + 1;
    }
}
```

79. 271. Encode and Decode Strings
[LeetCode](https://leetcode.com/problems/encode-and-decode-strings/description/)
[blog]()
出现面经：
[google被强制转为SETI咋办？](https://www.1point3acres.com/bbs/thread-211115-1-1.html)


thought:
要求给一个list的字符串，转成一个字符串之后还能再解码回来。

code：
```java
public class Codec {

    // Encodes a list of strings to a single string.
    public String encode(List<String> strs) {
        StringBuilder sb = new StringBuilder();
        //encoded string is in the format that [length + '/' + string]
        for (String s : strs){
            sb.append(s.length()).append('/').append(s);
        }
        return sb.toString();
    }

    // Decodes a single string to a list of strings.
    public List<String> decode(String s) {
        List<String> res = new ArrayList<>();
        int i = 0;
        while (i < s.length()){
            //find slash and split to a single string, then add to res.
            int slash = s.indexOf('/', i);
            int size = Integer.valueOf(s.substring(i, slash));
            res.add(s.substring(slash + 1, slash + size + 1));
            i = slash + size + 1;
        }
        return res;
    }
}

// Your Codec object will be instantiated and called as such:
// Codec codec = new Codec();
// codec.decode(codec.encode(strs));
```

80. 288. Unique Word Abbreviation
[LeetCode](https://leetcode.com/problems/unique-word-abbreviation/description/)
[blog]()
出现面经：
有不少提到过

thought:

code：
```java
public class ValidWordAbbr {
    private Map<String, String> map;
    
    public ValidWordAbbr(String[] dictionary) {
        map = new HashMap<>();
        for (String word : dictionary){
            String abbreviation = getAbbreviation(word);
            if (map.containsKey(abbreviation)){
                if (!map.get(abbreviation).equals(word)){
                    map.put(abbreviation, "");
                }
            }
            else{
                map.put(abbreviation, word);
            }
           
        }
    }
    
    public boolean isUnique(String word) {
        String abbreviation = getAbbreviation(word);
        if (!map.containsKey(abbreviation)){
            return true;
        }
        else{
            return map.get(abbreviation).equals(word);
        }
        
    }
    
    private String getAbbreviation(String word){
        StringBuilder sb = new StringBuilder();
        if (word.length() <= 2){
            return word;
        }
        else{
            sb.append(word.charAt(0)).append(Integer.toString(word.length() - 2)).append(word.charAt(word.length() - 1));
        }
        return sb.toString();
    }
}

/**
 * Your ValidWordAbbr object will be instantiated and called as such:
 * ValidWordAbbr obj = new ValidWordAbbr(dictionary);
 * boolean param_1 = obj.isUnique(word);
 */
```

37. 
[LeetCode]()
[blog]()
出现面经：
[Google Onsite 血跪](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=452151&ctid=84995)


thought:

code：
```java

```

37. 
[LeetCode]()
[blog]()
出现面经：
[Google Onsite 血跪](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=452151&ctid=84995)


thought:

code：
```java

```

37. 
[LeetCode]()
[blog]()
出现面经：
[Google Onsite 血跪](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=452151&ctid=84995)


thought:

code：
```java

```

37. 
[LeetCode]()
[blog]()
出现面经：
[Google Onsite 血跪](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=452151&ctid=84995)


thought:

code：
```java

```

37. 
[LeetCode]()
[blog]()
出现面经：
[Google Onsite 血跪](https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=452151&ctid=84995)


thought:

code：
```java

```


