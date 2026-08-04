"""Instructor-style DSA concept registry.
Replace it with your own course material at any time:

    python -m src.data.ingest --input path/to/your/notes --config configs/default.yaml


"""

from __future__ import annotations

from typing import Dict, List

RAW = r"""
## Arrays and Strings
- dynamic_array | dynamic array, resizable array, vector, ArrayList, amortized doubling
  idea: A dynamic array keeps a contiguous buffer with spare capacity so appends are usually free.
  formal: When the buffer is full, capacity is doubled and all n elements are copied; the cost of n appends is O(n), so each append is O(1) amortized.
  cost: index O(1); append O(1) amortized, O(n) worst case; insert or delete at position i O(n - i)
  pitfall: Students quote O(1) for append as a worst-case bound and forget the occasional O(n) resize.
- prefix_sums | prefix sum, cumulative sum, range sum query
  idea: Precomputing running totals turns any range-sum question into one subtraction.
  formal: With P[0] = 0 and P[i] = P[i-1] + a[i-1], the sum of a[l..r] equals P[r+1] - P[l].
  cost: build O(n); each range query O(1); extra space O(n)
  pitfall: Off-by-one errors from mixing inclusive and exclusive endpoints in P.
- two_pointers | two pointers, opposite ends, pair sum
  idea: Two indices moving toward each other replace a nested loop when the array is sorted.
  formal: If a is sorted and a[i] + a[j] > target then j must decrease, because every element left of j is smaller; this monotone argument proves correctness.
  cost: time O(n) after sorting; space O(1)
  pitfall: Applying the technique to an unsorted array, where the monotonicity argument fails.
- sliding_window | sliding window, subarray window, variable window
  idea: Keep a window that expands on the right and shrinks on the left while a condition holds.
  formal: Every index enters and leaves the window at most once, so the total work is O(n) even though the inner loop looks nested.
  cost: time O(n); space O(1) or O(k) with a frequency map
  pitfall: Shrinking with an if instead of a while, which leaves the window invalid.
- kadane | Kadane algorithm, maximum subarray, best contiguous sum
  idea: At each index decide whether to extend the previous best subarray or start fresh.
  formal: best[i] = max(a[i], best[i-1] + a[i]); the answer is max over i of best[i].
  cost: time O(n); space O(1)
  pitfall: Initialising the running maximum to 0, which breaks on all-negative arrays.
- array_rotation | rotate array, reversal algorithm, cyclic shift
  idea: Rotating by k is three reversals: reverse all, reverse the first k, reverse the rest.
  formal: Reversal is an involution, and reverse(reverse(A)reverse(B)) = BA, which is exactly the rotation.
  cost: time O(n); space O(1)
  pitfall: Forgetting to take k modulo n before rotating.
- kmp_matching | KMP, Knuth Morris Pratt, failure function, pattern matching
  idea: When a mismatch happens, the longest border of the matched prefix tells us how far we may safely jump.
  formal: lps[i] is the length of the longest proper prefix of the pattern that is also a suffix of pattern[0..i]; the text pointer never moves backwards.
  cost: preprocessing O(m); matching O(n + m); space O(m)
  pitfall: Recomputing lps inside the matching loop instead of once beforehand.
- rabin_karp | Rabin Karp, rolling hash, string hashing search
  idea: Hash the pattern once and roll the hash over the text one character at a time.
  formal: H(s[i+1..i+m]) = (H(s[i..i+m-1]) - s[i]*b^(m-1)) * b + s[i+m], all taken modulo a large prime.
  cost: expected O(n + m); worst case O(nm) under adversarial collisions
  pitfall: Skipping the character-by-character verification after a hash match.
- frequency_counting | character count, anagram check, frequency map
  idea: Two strings are anagrams exactly when their character multisets agree.
  formal: Counting into an array of size sigma and comparing counts decides anagram in linear time without sorting.
  cost: time O(n + sigma); space O(sigma)
  pitfall: Assuming a 26-letter alphabet when the input may contain digits or Unicode.
- subarray_sum_hashmap | subarray sum equals k, prefix sum with hash map
  idea: Store how many times each prefix sum has been seen; a match means a valid subarray ends here.
  formal: The number of subarrays ending at r with sum k equals the count of previously seen prefix values P[r+1] - k.
  cost: time O(n); space O(n)
  pitfall: Forgetting to seed the map with prefix sum 0 exactly once.
- matrix_traversal | 2D array, matrix traversal, spiral order, row major
  idea: A two-dimensional array is a one-dimensional block addressed by row times width plus column.
  formal: For row-major storage, element (i, j) of an m by n matrix lives at flat index i*n + j.
  cost: full traversal O(mn); random access O(1)
  pitfall: Swapping the row and column bounds in the loops, which silently works only for square matrices.
- difference_array | difference array, range update, offline updates
  idea: To add v over a range, record +v at the start and -v just past the end, then take a prefix sum at the end.
  formal: With D[l] += v and D[r+1] -= v, the prefix sum of D reconstructs the updated array.
  cost: each range update O(1); final reconstruction O(n)
  pitfall: Writing to index r instead of r+1, which shortens every update by one cell.

## Linked Lists
- singly_linked_list | singly linked list, node, next pointer, head
  idea: Each node stores a value and a reference to the next node, so memory need not be contiguous.
  formal: The list is defined recursively as either null or a node whose next field is a list; the head reference owns the whole chain.
  cost: access by position O(n); insert or delete given the node O(1)
  pitfall: Losing the rest of the list by reassigning next before saving it in a temporary variable.
- list_traversal | traverse linked list, walk the list, iterate nodes
  idea: Walk with a cursor that advances until it becomes null.
  formal: The loop invariant is that the cursor points to an unvisited node and every node before it has been processed exactly once.
  cost: time O(n); space O(1)
  pitfall: Dereferencing the cursor after the loop, when it is already null.
- list_insertion | insert into linked list, insert at head, insert after node
  idea: Insertion is two pointer writes in the right order.
  formal: To insert x after p: x.next = p.next, then p.next = x; reversing these two statements loses the tail.
  cost: at head O(1); at position k O(k)
  pitfall: Handling the empty-list case separately instead of using a dummy head.
- list_deletion | delete node, remove from linked list, unlink
  idea: To delete a node you need its predecessor, so keep a trailing pointer.
  formal: With prev.next = target.next the target becomes unreachable and can be freed.
  cost: given the predecessor O(1); by value O(n)
  pitfall: Freeing the node before reading its next field.
- doubly_linked_list | doubly linked list, prev pointer, bidirectional list
  idea: Storing both next and prev makes deletion possible from the node alone.
  formal: The invariant node.next.prev == node must hold for every non-tail node after any update.
  cost: insert or delete given the node O(1); extra memory one pointer per node
  pitfall: Updating only one of the two directions, which corrupts backward traversal.
- circular_list | circular linked list, ring buffer list, round robin
  idea: The tail points back to the head, so traversal never hits null.
  formal: Termination must be tested against the starting node rather than against null.
  cost: traversal O(n); insert after tail O(1)
  pitfall: Writing a while (cur != null) loop, which never ends.
- reverse_iterative | reverse linked list iteratively, three pointer reversal
  idea: Walk once, flipping each next pointer backwards while remembering the following node.
  formal: With prev, cur, nxt the body is nxt = cur.next; cur.next = prev; prev = cur; cur = nxt; the answer is prev.
  cost: time O(n); space O(1)
  pitfall: Returning cur, which is null, instead of prev.
- reverse_recursive | reverse linked list recursively, recursive reversal
  idea: Reverse the tail, then hook the current node onto the end of the reversed part.
  formal: rev(head) returns the new head; the fix-up is head.next.next = head followed by head.next = null.
  cost: time O(n); space O(n) for the call stack
  pitfall: Forgetting head.next = null, which creates a two-node cycle.
- floyd_cycle | cycle detection, Floyd tortoise and hare, loop in linked list
  idea: A fast pointer moving two steps meets a slow pointer inside any cycle.
  formal: If the cycle has length c and starts at distance mu, the pointers meet after at most mu + c steps; restarting one pointer at the head makes them meet at the cycle entry.
  cost: time O(n); space O(1)
  pitfall: Advancing fast by two without first checking that fast and fast.next are non-null.
- middle_node | find middle, slow fast pointers, midpoint of list
  idea: When the fast pointer reaches the end, the slow pointer is halfway.
  formal: After k iterations slow has moved k and fast 2k, so slow sits at floor(n/2) when fast exits.
  cost: time O(n); space O(1)
  pitfall: Ambiguity for even lengths, where the two middle definitions differ by one node.
- merge_sorted_lists | merge two sorted lists, merge linked lists
  idea: Repeatedly take the smaller head and append it to the result.
  formal: The merge is stable and uses only pointer rewiring, so no new nodes are allocated.
  cost: time O(n + m); space O(1) with iteration
  pitfall: Forgetting to attach the non-empty remainder after one list runs out.
- lru_cache | LRU cache, hash map plus doubly linked list
  idea: A hash map gives O(1) lookup while a doubly linked list keeps recency order.
  formal: Every access moves its node to the front; eviction removes the tail node and its map entry together.
  cost: get O(1); put O(1)
  pitfall: Removing the node from the list but leaving a stale key in the map.

## Stacks and Queues
- stack_adt | stack, LIFO, push pop peek
  idea: A stack serves the most recently added item first.
  formal: push, pop and peek are all O(1) whether the stack is backed by an array or a linked list.
  cost: push O(1) amortized on arrays, O(1) on lists; pop O(1); peek O(1)
  pitfall: Calling pop on an empty stack instead of checking isEmpty first.
- balanced_parentheses | balanced brackets, parenthesis matching, valid parentheses
  idea: Push every opener and require the matching opener on top when a closer arrives.
  formal: The string is balanced exactly when the stack never underflows and is empty at the end.
  cost: time O(n); space O(n)
  pitfall: Returning true without checking that the stack is empty at the end.
- infix_to_postfix | shunting yard, infix to postfix, expression conversion
  idea: Operands go straight out; operators wait on a stack until a lower-precedence one arrives.
  formal: Pop while the operator on top has higher precedence, or equal precedence with left associativity.
  cost: time O(n); space O(n)
  pitfall: Treating right-associative exponentiation with the same rule as left-associative operators.
- postfix_evaluation | evaluate postfix, reverse polish notation
  idea: Push operands and, on each operator, pop two operands and push the result.
  formal: The operand popped first is the right-hand argument, which matters for subtraction and division.
  cost: time O(n); space O(n)
  pitfall: Reversing the operand order for non-commutative operators.
- monotonic_stack | monotonic stack, next greater element, previous smaller
  idea: Keep the stack sorted so each element is popped by its answer.
  formal: Every index is pushed once and popped once, giving O(n) total work despite the inner while loop.
  cost: time O(n); space O(n)
  pitfall: Storing values instead of indices, which loses the position needed for widths.
- min_stack | min stack, stack with getMin, auxiliary stack
  idea: Store the running minimum alongside each element.
  formal: Pushing the pair (value, min(value, currentMin)) makes getMin an O(1) read of the top.
  cost: all operations O(1); space O(n)
  pitfall: Updating the minimum on push but not restoring it on pop.
- queue_adt | queue, FIFO, enqueue dequeue
  idea: A queue serves items in arrival order.
  formal: With separate front and rear indices both enqueue and dequeue are O(1).
  cost: enqueue O(1); dequeue O(1)
  pitfall: Implementing dequeue as a shift of the whole array, which degrades to O(n).
- circular_queue | circular queue, ring buffer, wrap around queue
  idea: Wrapping indices with modulo reuses the space freed at the front.
  formal: rear = (rear + 1) mod capacity; the queue is full when (rear + 1) mod capacity == front if one slot is kept empty.
  cost: enqueue O(1); dequeue O(1); space O(capacity)
  pitfall: Confusing the full and empty conditions when the sentinel slot is not reserved.
- deque | deque, double ended queue, sliding window maximum
  idea: A deque allows push and pop at both ends, which powers sliding-window extremes.
  formal: Keeping the deque decreasing lets the front always hold the window maximum in O(1).
  cost: each operation O(1) amortized; window maximum over n elements O(n)
  pitfall: Forgetting to drop indices that have fallen out of the window.
- queue_from_stacks | queue using two stacks, amortized transfer
  idea: One stack takes input, the other serves output, and elements move only when the output side is empty.
  formal: Each element is pushed and popped at most twice, so dequeue is O(1) amortized.
  cost: enqueue O(1); dequeue O(1) amortized, O(n) worst case
  pitfall: Transferring on every dequeue, which makes the cost O(n) each time.
- stack_from_queues | stack using two queues, rotation trick
  idea: Rotate the queue after each push so the newest element sits at the front.
  formal: Push costs O(n) rotations while pop is O(1), or the costs can be swapped.
  cost: one operation O(1) and the other O(n), depending on the variant chosen
  pitfall: Claiming both operations are O(1), which is impossible with plain queues.
- call_stack | call stack, recursion stack, stack frame
  idea: Recursion is a stack the runtime manages for you.
  formal: Each frame stores parameters, locals and the return address; depth d costs O(d) memory.
  cost: space O(depth); constant-time push and pop per call
  pitfall: Ignoring recursion depth in the space complexity of an otherwise O(1)-space algorithm.

## Trees and BSTs
- tree_terminology | tree terms, root leaf height depth, binary tree
  idea: A tree is a connected acyclic graph in which every node except the root has one parent.
  formal: A binary tree of height h has at most 2^(h+1) - 1 nodes, and n nodes force height at least floor(log2 n).
  cost: n nodes have exactly n - 1 edges
  pitfall: Mixing up height, which counts edges downward, with depth, which counts edges to the root.
- dfs_traversals | inorder preorder postorder, depth first traversal of tree
  idea: The three orders differ only in when the node itself is visited relative to its subtrees.
  formal: Inorder on a BST emits keys in sorted order, which is the standard correctness test.
  cost: time O(n); space O(h) for the stack
  pitfall: Believing preorder alone reconstructs a tree, which needs a second traversal or null markers.
- level_order | level order traversal, BFS on tree, breadth first tree
  idea: A queue visits nodes level by level.
  formal: Processing the queue in batches of its current size separates the levels cleanly.
  cost: time O(n); space O(w) where w is the maximum width
  pitfall: Using a stack, which turns the level order into a depth-first order.
- tree_height | height of tree, maximum depth, recursive height
  idea: The height of a node is one more than the taller of its two children.
  formal: height(null) = -1 with edge counting, or 0 with node counting; the convention must be fixed before writing recurrences.
  cost: time O(n); space O(h)
  pitfall: Mixing the two height conventions inside a single balance check.
- bst_property | binary search tree property, BST invariant, ordered tree
  idea: Everything in the left subtree is smaller and everything in the right subtree is larger.
  formal: The invariant is global, not local: it constrains a node against every ancestor, not just its parent.
  cost: search, insert and delete O(h), which is O(log n) when balanced and O(n) when degenerate
  pitfall: Checking only parent-child pairs, which wrongly accepts an invalid tree.
- bst_search_insert | BST search, BST insert, tree lookup
  idea: Compare with the current key and descend left or right accordingly.
  formal: The search path is unique, and insertion always attaches a new leaf at the point where the search fails.
  cost: O(h) time; O(1) extra space when written iteratively
  pitfall: Inserting duplicates without a policy, which breaks later deletions.
- bst_deletion | BST delete, remove node from BST, two child case
  idea: Deleting a node with two children means replacing it with its inorder successor.
  formal: The successor is the leftmost node of the right subtree and has at most one child, so its own removal is easy.
  cost: O(h) time
  pitfall: Copying the successor value but forgetting to delete the successor node itself.
- successor_predecessor | inorder successor, predecessor, next larger key
  idea: The successor is the leftmost node on the right, or the lowest ancestor turning left.
  formal: If the right subtree is empty, the successor is the last ancestor from which we descended left.
  cost: O(h) with parent pointers or an ancestor stack
  pitfall: Handling only the right-subtree case and returning null for the ancestor case.
- validate_bst | validate BST, check BST, range check
  idea: Pass an allowed interval down the recursion instead of comparing neighbours.
  formal: Each recursive call checks low < key < high and narrows the interval for its children.
  cost: time O(n); space O(h)
  pitfall: Using inclusive bounds when duplicates are disallowed.
- lca | lowest common ancestor, LCA, common parent
  idea: In a BST the split point where the two keys diverge is their lowest common ancestor.
  formal: Descend while both keys are on the same side; the first node between them is the answer.
  cost: BST O(h); general binary tree O(n)
  pitfall: Applying the BST shortcut to an unordered binary tree.
- avl_rotations | AVL tree, rotation, balance factor, self balancing
  idea: Restore balance with a local rotation whenever a subtree gets two levels out of step.
  formal: The balance factor stays in {-1, 0, +1}; LL and RR need one rotation, LR and RL need two.
  cost: search, insert and delete O(log n); rotations O(1) each
  pitfall: Fixing the balance factors of the wrong nodes after a double rotation.
- red_black | red black tree, RB tree, balanced BST properties
  idea: Colour constraints keep the longest path at most twice the shortest.
  formal: Every path from a node to its leaves contains the same number of black nodes, giving height O(log n).
  cost: search, insert and delete O(log n)
  pitfall: Assuming red-black trees are height balanced like AVL trees; they are only black-height balanced.
- segment_tree | segment tree, range query tree, interval tree
  idea: Each node stores an aggregate of a contiguous range so queries touch only O(log n) nodes.
  formal: A range decomposes into at most 2 log n canonical nodes; point update walks one root-to-leaf path.
  cost: build O(n); query O(log n); update O(log n)
  pitfall: Sizing the array as 2n instead of 4n, which overflows for non-power-of-two n.
- trie | trie, prefix tree, dictionary tree, autocomplete
  idea: Shared prefixes are stored once along the path from the root.
  formal: Lookup cost depends on the key length L, not on the number of stored keys.
  cost: insert and search O(L); space O(total characters * sigma) in the array form
  pitfall: Forgetting the end-of-word flag, which makes every prefix look like a stored word.

## Heaps and Priority Queues
- binary_heap | binary heap, array heap, complete binary tree
  idea: A heap is a complete binary tree stored in an array with no pointers.
  formal: For zero-based indexing, children of i are 2i+1 and 2i+2 and the parent is floor((i-1)/2).
  cost: peek O(1); insert O(log n); extract O(log n)
  pitfall: Using one-based index formulas on a zero-based array.
- heap_insert | sift up, percolate up, heap insert
  idea: Place the new element at the end and bubble it up while it beats its parent.
  formal: The path from a leaf to the root has length log n, bounding the number of swaps.
  cost: O(log n) time; O(1) extra space
  pitfall: Bubbling up from the root instead of from the newly appended leaf.
- heap_extract | sift down, heapify down, extract min
  idea: Move the last element to the root and sink it until the heap order is restored.
  formal: Sinking compares with the better of the two children, so at most two comparisons per level.
  cost: O(log n)
  pitfall: Comparing with only the left child, which breaks the heap order.
- build_heap | build heap, heapify array, Floyd construction
  idea: Sift down from the last internal node backwards to the root.
  formal: The sum over levels of nodes times height converges, giving O(n) rather than O(n log n).
  cost: O(n) time; O(1) space
  pitfall: Repeated insertion, which costs O(n log n) instead of the linear build.
- heapsort | heap sort, in place sort with heap
  idea: Build a max-heap and repeatedly swap the root to the end of the array.
  formal: The sorted suffix grows by one on each of the n-1 extractions.
  cost: time O(n log n) always; space O(1); not stable
  pitfall: Forgetting to shrink the heap size after each swap.
- priority_queue_adt | priority queue, PQ ADT, key ordering
  idea: A priority queue serves the best element rather than the oldest.
  formal: A heap gives O(log n) insert and extract; a sorted array gives O(1) extract but O(n) insert.
  cost: heap-backed insert O(log n), extract O(log n), peek O(1)
  pitfall: Assuming a priority queue iterates in sorted order, which a heap does not.
- k_largest | top k elements, k largest, selection with heap
  idea: Keep a min-heap of size k and evict the smallest whenever a bigger element arrives.
  formal: Only k elements are ever stored, so memory is independent of the stream length.
  cost: time O(n log k); space O(k)
  pitfall: Using a max-heap of size k, which cannot cheaply drop the weakest survivor.
- running_median | median of stream, two heaps, running median
  idea: A max-heap holds the lower half and a min-heap the upper half.
  formal: Rebalancing keeps the sizes within one, so the median is the top of the larger heap.
  cost: insert O(log n); median query O(1)
  pitfall: Letting the size difference exceed one before rebalancing.
- d_ary_heap | d-ary heap, multiway heap, cache friendly heap
  idea: Giving each node d children makes the tree shallower and insertions cheaper.
  formal: Height becomes log_d n, so insert is O(log_d n) while extract costs O(d log_d n).
  cost: insert O(log_d n); extract O(d log_d n)
  pitfall: Increasing d to speed up extraction, which actually slows it down.
- heap_in_graph | heap in Dijkstra, priority queue in Prim, lazy deletion
  idea: Graph algorithms use a priority queue to always expand the cheapest frontier node.
  formal: With a binary heap Dijkstra runs in O((V + E) log V); lazy deletion pushes duplicates and skips stale pops.
  cost: O((V + E) log V) with a binary heap
  pitfall: Not skipping stale heap entries whose recorded distance is worse than the settled one.
- decrease_key | decrease key, update priority, indexed heap
  idea: Lowering a key requires knowing where the element currently sits.
  formal: An index map from element to heap position makes decrease-key O(log n); without it the search is O(n).
  cost: decrease-key O(log n) with an index map
  pitfall: Forgetting to update the index map after every swap inside sift operations.

## Hashing
- hash_function | hash function, hashing basics, hash code
  idea: A hash function maps a key to a bucket index, ideally spreading keys evenly.
  formal: Good hashes are deterministic, fast and avalanche-like, so a one-bit key change scrambles the output.
  cost: hashing a key of length L costs O(L)
  pitfall: Treating O(1) hashing as free for long string keys.
- load_factor | load factor, alpha, table occupancy
  idea: The load factor is the ratio of stored entries to buckets and drives the expected probe count.
  formal: With chaining the expected chain length is alpha; with linear probing the expected probes grow like 1/(1-alpha).
  cost: keep alpha below about 0.75 for open addressing
  pitfall: Letting the table fill up without rehashing, so lookups degrade toward O(n).
- separate_chaining | chaining, bucket list, collision resolution by chaining
  idea: Each bucket holds a list of all keys that hash there.
  formal: Under simple uniform hashing, expected search time is O(1 + alpha).
  cost: expected O(1); worst case O(n) when all keys collide
  pitfall: Assuming worst-case O(1), which an adversarial key set defeats.
- linear_probing | open addressing, linear probing, primary clustering
  idea: On a collision, walk forward to the next free slot.
  formal: Probe sequence h(k, i) = (h(k) + i) mod m; deleted slots need tombstones to keep probe chains intact.
  cost: expected O(1) at low load; degrades sharply as alpha approaches 1
  pitfall: Deleting by clearing a slot, which cuts probe chains and hides later keys.
- double_hashing | quadratic probing, double hashing, secondary clustering
  idea: Vary the probe step so colliding keys do not follow the same path.
  formal: h(k, i) = (h1(k) + i * h2(k)) mod m, where h2 must never return 0 and should be coprime with m.
  cost: expected O(1) with better clustering behaviour than linear probing
  pitfall: Choosing h2 that shares a factor with m, so only part of the table is reachable.
- rehashing | rehashing, table resize, dynamic hash table
  idea: When the load factor crosses a threshold, allocate a bigger table and reinsert everything.
  formal: Doubling makes the amortized insert O(1) by the same argument as the dynamic array.
  cost: rehash O(n) occasionally; insert O(1) amortized
  pitfall: Copying buckets without recomputing indices for the new table size.
- collision_probability | birthday paradox, collision probability, hash collisions
  idea: Collisions appear far earlier than intuition suggests.
  formal: With m buckets, a collision becomes likely once about sqrt(m) keys have been inserted.
  cost: expected first collision after about 1.25 * sqrt(m) insertions
  pitfall: Sizing a table by the key count alone and ignoring collision probability.
- hash_map_ops | hash map, dictionary operations, average complexity
  idea: A hash map trades ordering for expected constant-time access.
  formal: get, put and delete are O(1) expected but unordered; a balanced tree gives O(log n) with order.
  cost: expected O(1) per operation; O(n) worst case
  pitfall: Using a hash map when the task also requires sorted iteration or range queries.
- hash_set | hash set, set membership, deduplication
  idea: A set is a map that stores only keys, used for membership and de-duplication.
  formal: Membership testing turns many O(n^2) scans into O(n) passes.
  cost: add and contains expected O(1); space O(n)
  pitfall: Relying on insertion order, which a plain hash set does not guarantee.
- universal_hashing | universal hashing, randomized hash family, adversarial keys
  idea: Picking the hash function at random defeats worst-case key sets.
  formal: A family is universal when the collision probability for any two distinct keys is at most 1/m.
  cost: expected O(1) per operation regardless of input distribution
  pitfall: Fixing one hash function in code, which is vulnerable to crafted collision attacks.
- string_hashing | polynomial hashing, string hash, modulus choice
  idea: Treat the string as digits of a number in a chosen base.
  formal: H(s) = sum of s[i] * b^(n-1-i) modulo a large prime; double hashing with two moduli makes collisions negligible.
  cost: precompute O(n); substring hash O(1)
  pitfall: Using a small modulus, which makes anagram-style collisions common.

## Graphs and Traversals
- graph_representation | adjacency list, adjacency matrix, graph storage
  idea: Store neighbours per vertex in a list, or all pairs in a matrix.
  formal: The list uses O(V + E) space and iterates neighbours in O(deg v); the matrix uses O(V^2) but answers edge queries in O(1).
  cost: list O(V + E) space; matrix O(V^2) space
  pitfall: Using a matrix for a sparse graph, where the memory dwarfs the edge count.
- graph_terminology | degree, path, cycle, connected graph
  idea: Vertices, edges, degrees, paths and cycles form the vocabulary every graph algorithm assumes.
  formal: The sum of all degrees equals twice the number of edges in an undirected graph.
  cost: a simple graph on V vertices has at most V(V-1)/2 edges
  pitfall: Applying undirected reasoning about degrees to a directed graph.
- bfs | breadth first search, BFS, shortest path unweighted
  idea: Explore all neighbours at the current distance before going deeper.
  formal: BFS from s assigns dist[v] = dist[u] + 1 on first discovery, which is provably the shortest hop count.
  cost: time O(V + E); space O(V)
  pitfall: Marking a vertex visited when it is popped rather than when it is pushed, which enqueues duplicates.
- dfs | depth first search, DFS, recursive graph traversal
  idea: Follow one branch as deep as possible, then backtrack.
  formal: DFS classifies edges as tree, back, forward or cross; a back edge is exactly a cycle witness.
  cost: time O(V + E); space O(V)
  pitfall: Running DFS from only one source on a disconnected graph.
- connected_components | connected components, islands, component count
  idea: Restart the traversal from every unvisited vertex and count the restarts.
  formal: Each traversal marks exactly one component, so the number of restarts is the component count.
  cost: time O(V + E)
  pitfall: Resetting the visited array between restarts, which recounts old components.
- cycle_detection | detect cycle, back edge, directed cycle
  idea: In an undirected graph a visited non-parent neighbour signals a cycle; in a directed graph the vertex must still be on the recursion stack.
  formal: Directed cycle detection uses three colours: white unvisited, grey on stack, black finished; a grey neighbour is a cycle.
  cost: time O(V + E)
  pitfall: Using the undirected parent rule on a directed graph, which reports false cycles.
- topological_sort | topological order, Kahn algorithm, DAG ordering
  idea: Repeatedly output a vertex with no remaining incoming edges.
  formal: A topological order exists if and only if the graph is a DAG; Kahn's queue empties early exactly when a cycle remains.
  cost: time O(V + E); space O(V)
  pitfall: Concluding a valid order exists without checking that all V vertices were emitted.
- dijkstra | Dijkstra, shortest path weighted, greedy shortest path
  idea: Always settle the unvisited vertex with the smallest tentative distance.
  formal: Correct only for non-negative weights, because settling assumes no later path can be cheaper.
  cost: O((V + E) log V) with a binary heap
  pitfall: Running it on a graph with negative edges and trusting the result.
- bellman_ford | Bellman Ford, negative weights, edge relaxation
  idea: Relax every edge V-1 times, which is enough for any shortest path.
  formal: A shortest path has at most V-1 edges; a further successful relaxation proves a negative cycle.
  cost: time O(VE); space O(V)
  pitfall: Stopping after V-1 rounds without the extra pass that detects negative cycles.
- floyd_warshall | Floyd Warshall, all pairs shortest path, DP on graphs
  idea: Allow one more intermediate vertex at a time and improve every pair.
  formal: d[k][i][j] = min(d[k-1][i][j], d[k-1][i][k] + d[k-1][k][j]), collapsible to a 2D array.
  cost: time O(V^3); space O(V^2)
  pitfall: Putting the k loop innermost, which breaks the dynamic-programming order.
- kruskal | Kruskal, minimum spanning tree, sort edges
  idea: Add the cheapest edge that does not create a cycle.
  formal: The cut property guarantees that the lightest edge crossing any cut belongs to some MST.
  cost: O(E log E) dominated by sorting, with union-find for cycle checks
  pitfall: Detecting cycles with a traversal per edge instead of union-find.
- prim | Prim, MST from a vertex, grow the tree
  idea: Grow one tree by repeatedly attaching the cheapest edge leaving it.
  formal: Also justified by the cut property, with the cut being tree versus non-tree vertices.
  cost: O((V + E) log V) with a binary heap
  pitfall: Adding the cheapest global edge rather than the cheapest edge incident to the current tree.
- union_find | disjoint set union, union find, DSU, path compression
  idea: Keep a forest where each set has a representative root.
  formal: With union by rank and path compression, m operations cost O(m alpha(n)) where alpha is the inverse Ackermann function.
  cost: near-constant amortized per operation
  pitfall: Comparing parent[x] == parent[y] instead of comparing the two roots found by find.
- bipartite_check | bipartite graph, two colouring, odd cycle
  idea: Two-colour the graph during BFS and fail on a same-coloured edge.
  formal: A graph is bipartite if and only if it contains no odd-length cycle.
  cost: time O(V + E)
  pitfall: Testing only the component containing the start vertex.
- scc_kosaraju | strongly connected components, Kosaraju, Tarjan, SCC
  idea: Order vertices by finish time, then explore the reversed graph in that order.
  formal: The reversed graph has the same SCCs, and finish-time order visits them in reverse topological order.
  cost: time O(V + E) with two DFS passes
  pitfall: Reusing the first pass's visited array in the second pass.
- grid_as_graph | grid graph, maze traversal, flood fill
  idea: Grid cells are vertices and adjacent cells are edges, so BFS and DFS apply directly.
  formal: A cell (r, c) has up to four neighbours; BFS gives the shortest number of moves in an unweighted maze.
  cost: time O(rows * cols)
  pitfall: Forgetting bounds checks before indexing a neighbour cell.

## Sorting Algorithms
- bubble_sort | bubble sort, adjacent swaps
  idea: Repeatedly swap adjacent out-of-order pairs so the largest element bubbles to the end.
  formal: After pass k the last k elements are final; an early exit when no swap occurs makes the best case O(n).
  cost: time O(n^2) average and worst, O(n) best with the swap flag; space O(1); stable
  pitfall: Omitting the early-exit flag and then claiming a best case of O(n).
- selection_sort | selection sort, minimum selection
  idea: Repeatedly pick the smallest remaining element and place it next.
  formal: Exactly n-1 swaps are performed regardless of the input order.
  cost: time O(n^2) always; space O(1); not stable
  pitfall: Expecting selection sort to be faster on nearly sorted input; it is not.
- insertion_sort | insertion sort, shift and insert
  idea: Grow a sorted prefix by inserting each new element into its place.
  formal: The running time is O(n + inversions), which is why it is used for small or nearly sorted arrays.
  cost: time O(n^2) worst, O(n) best; space O(1); stable
  pitfall: Using a linear scan plus a swap loop and calling it binary insertion sort.
- merge_sort | merge sort, divide and conquer sort, stable sort
  idea: Split in half, sort both halves, then merge them in linear time.
  formal: T(n) = 2T(n/2) + O(n), which the master theorem solves as O(n log n).
  cost: time O(n log n) always; space O(n); stable
  pitfall: Allocating a fresh buffer inside every merge instead of reusing one.
- quick_sort | quicksort, partition, pivot
  idea: Partition around a pivot so everything smaller is left and everything larger is right.
  formal: T(n) = T(k) + T(n-k-1) + O(n); random or median-of-three pivots make O(n log n) expected.
  cost: expected O(n log n); worst case O(n^2); space O(log n) for recursion; not stable
  pitfall: Always choosing the first element as pivot, which makes sorted input quadratic.
- quickselect | quickselect, kth smallest, selection algorithm
  idea: Partition like quicksort but recurse into only the side that contains the target rank.
  formal: The expected cost is n + n/2 + n/4 + ... = O(n).
  cost: expected O(n); worst case O(n^2)
  pitfall: Recursing into both partitions, which turns it back into a full sort.
- heap_sort_alg | heapsort algorithm, sort with heap, in place n log n
  idea: Build a max-heap and swap the root with the shrinking tail.
  formal: Build costs O(n) and the n-1 extractions cost O(n log n) in total.
  cost: time O(n log n); space O(1); not stable
  pitfall: Assuming heapsort is stable because it is comparison based.
- counting_sort | counting sort, non comparison sort, key counting
  idea: Count occurrences of each key and write them back in order.
  formal: With keys in [0, k) the running time is O(n + k), which beats n log n only when k is small.
  cost: time O(n + k); space O(n + k); stable when built with prefix counts
  pitfall: Using it for large key ranges, where the count array dominates memory.
- radix_sort | radix sort, LSD sort, digit by digit
  idea: Sort by each digit from least significant to most, using a stable sort per pass.
  formal: With d digits in base b the cost is O(d (n + b)); stability of the inner sort is essential.
  cost: time O(d (n + b)); space O(n + b)
  pitfall: Using an unstable inner sort, which destroys the ordering from earlier passes.
- bucket_sort | bucket sort, distribution sort
  idea: Scatter values into buckets by range, sort each bucket, then concatenate.
  formal: Uniformly distributed input gives O(n) expected time; skewed input collapses to one bucket.
  cost: expected O(n); worst case O(n^2)
  pitfall: Assuming linear time without checking that the data really is uniform.
- sort_stability | stable sort, stability, tie breaking
  idea: A stable sort preserves the original order of equal keys.
  formal: Stability is what makes multi-key sorting work by sorting on the least significant key first.
  cost: merge, insertion and counting sort are stable; quicksort and heapsort are not
  pitfall: Chaining two unstable sorts and expecting the first key order to survive.
- comparison_lower_bound | lower bound n log n, decision tree, comparison sorting
  idea: Any comparison sort must distinguish n! orderings using binary decisions.
  formal: A decision tree with n! leaves has height at least log2(n!) = Omega(n log n).
  cost: no comparison sort can beat Omega(n log n) in the worst case
  pitfall: Citing counting sort as a counterexample; it is not comparison based.
- hybrid_sorts | Timsort, introsort, hybrid sorting in libraries
  idea: Real library sorts switch strategy by input size and recursion depth.
  formal: Introsort starts with quicksort and falls back to heapsort past a depth limit; Timsort merges natural runs and uses insertion sort on small runs.
  cost: O(n log n) worst case with excellent constants
  pitfall: Benchmarking a textbook quicksort against a library sort and blaming the algorithm rather than the engineering.

## Searching Algorithms
- linear_search | linear search, sequential search
  idea: Scan until the target is found or the data ends.
  formal: On average n/2 comparisons are needed for a successful search in unsorted data.
  cost: time O(n); space O(1); no preprocessing
  pitfall: Sorting first just to binary search once, which costs more than a single scan.
- binary_search | binary search, halving search, sorted lookup
  idea: Compare with the middle element and discard half of the range.
  formal: The invariant is that the answer, if present, lies inside [lo, hi]; use mid = lo + (hi - lo) / 2 to avoid overflow.
  cost: time O(log n); space O(1)
  pitfall: Writing mid = (lo + hi) / 2 and updating bounds inconsistently, causing an infinite loop.
- binary_search_answer | binary search on answer, parametric search, monotone predicate
  idea: If feasibility is monotone in the answer, binary search the answer space itself.
  formal: Define ok(x) that is false then true; the boundary is found in O(log range) calls to ok.
  cost: O(log range * cost of ok)
  pitfall: Applying it when the predicate is not monotone, so the boundary is not unique.
- lower_upper_bound | lower bound, upper bound, first occurrence
  idea: Modified binary searches locate the first and last positions of a value.
  formal: Lower bound returns the first index with a[i] >= x; upper bound returns the first with a[i] > x, and their gap is the frequency.
  cost: O(log n) each
  pitfall: Returning immediately on a match, which finds some occurrence rather than the first.
- ternary_search | ternary search, unimodal function, peak finding
  idea: On a unimodal function, two probes discard one third of the range.
  formal: Comparing f(m1) and f(m2) removes the side that cannot contain the extremum.
  cost: O(log_{3/2} n) evaluations
  pitfall: Using it on a function that is not unimodal, where the discard rule is invalid.
- interpolation_search | interpolation search, guess position, uniform keys
  idea: Guess the position proportionally instead of always probing the middle.
  formal: On uniformly distributed keys the expected cost is O(log log n), but skewed data degrades it to O(n).
  cost: expected O(log log n); worst case O(n)
  pitfall: Dividing by zero when the low and high key values are equal.
- exponential_search | exponential search, galloping search, unbounded search
  idea: Double the bound until it overshoots, then binary search inside that window.
  formal: Finding the window costs O(log i) where i is the answer index, and the binary search costs the same again.
  cost: O(log i), better than O(log n) when the target is near the front
  pitfall: Starting the doubling at 0 instead of 1, which never advances.
- rotated_array_search | search in rotated sorted array, pivot search
  idea: One half of any split is still sorted, so decide which half can contain the target.
  formal: Compare a[lo] with a[mid] to identify the sorted half, then test whether the target lies inside its range.
  cost: O(log n) without duplicates; O(n) worst case with duplicates
  pitfall: Ignoring duplicates, which make the sorted-half test ambiguous.
- search_structure_choice | which data structure to search, hash vs tree vs array
  idea: The right search structure depends on whether you need order, range queries or raw speed.
  formal: Hash maps give expected O(1) point lookup; balanced BSTs give O(log n) with order; sorted arrays give O(log n) with the least memory.
  cost: choose by access pattern, not by asymptotics alone
  pitfall: Reaching for a hash map when the workload is dominated by range queries.

## Dynamic Programming (Intro)
- dp_principles | overlapping subproblems, optimal substructure, when DP applies
  idea: DP applies when the same subproblems recur and optimal solutions are built from optimal sub-solutions.
  formal: Without optimal substructure the recurrence is invalid; without overlap, plain divide and conquer is already enough.
  cost: total cost = number of states * transition cost
  pitfall: Applying DP to problems whose subproblems are independent, where recursion suffices.
- memo_vs_tab | memoization, tabulation, top down bottom up
  idea: Memoization caches a recursion; tabulation fills a table in dependency order.
  formal: Both explore the same state space; tabulation avoids stack depth, memoization skips unreachable states.
  cost: same asymptotics; memoization pays call overhead, tabulation may compute unused states
  pitfall: Tabulating in an order that reads a cell before it has been written.
- dp_fibonacci | fibonacci DP, memoized fibonacci, linear recurrence
  idea: Caching turns exponential recursion into a linear scan.
  formal: F(n) = F(n-1) + F(n-2) needs only the last two values, so space drops to O(1).
  cost: time O(n); space O(1) with rolling variables
  pitfall: Keeping the whole array when only two previous values matter.
- climbing_stairs | climbing stairs, step counting, count ways
  idea: The number of ways to reach step n is the sum of the ways to reach the steps you can jump from.
  formal: ways[i] = ways[i-1] + ways[i-2] with ways[0] = 1 and ways[1] = 1.
  cost: time O(n); space O(1)
  pitfall: Setting ways[0] = 0, which makes every later count wrong.
- knapsack_01 | 0/1 knapsack, bounded knapsack, item selection
  idea: For each item decide to take it or skip it, tracking the remaining capacity.
  formal: dp[i][w] = max(dp[i-1][w], value[i] + dp[i-1][w - weight[i]]) when weight[i] <= w.
  cost: time O(nW); space O(nW), reducible to O(W)
  pitfall: In the 1D version, iterating capacity upward, which reuses an item more than once.
- unbounded_knapsack | unbounded knapsack, coin change, repeated items
  idea: Items may be reused, so transitions read the current row rather than the previous one.
  formal: dp[w] = min over coins c of dp[w - c] + 1, with the capacity loop running upward.
  cost: time O(nW); space O(W)
  pitfall: Copying the 0/1 loop direction, which forbids reuse and gives the wrong answer.
- lcs | longest common subsequence, LCS, sequence alignment
  idea: Match the last characters, or drop one character from either string.
  formal: dp[i][j] = dp[i-1][j-1] + 1 if the characters match, else max(dp[i-1][j], dp[i][j-1]).
  cost: time O(nm); space O(nm), reducible to O(min(n, m))
  pitfall: Confusing subsequence, which allows gaps, with substring, which does not.
- lis | longest increasing subsequence, LIS, patience sorting
  idea: Either extend the best chain ending earlier, or start a new one.
  formal: The O(n^2) recurrence is dp[i] = 1 + max over j < i with a[j] < a[i]; binary search over tail values gives O(n log n).
  cost: O(n^2) simple, O(n log n) with binary search
  pitfall: Reading the tails array as an actual subsequence; it only records lengths.
- edit_distance | edit distance, Levenshtein, string alignment
  idea: Transform one string into another with insertions, deletions and substitutions.
  formal: dp[i][j] = dp[i-1][j-1] if characters match, else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]).
  cost: time O(nm); space O(nm), reducible to O(min(n, m))
  pitfall: Initialising the first row and column to zero instead of to the index.
- matrix_chain | matrix chain multiplication, parenthesisation, interval DP
  idea: Choose the split point that minimises the cost of multiplying a chain of matrices.
  formal: dp[i][j] = min over k of dp[i][k] + dp[k+1][j] + p[i-1]*p[k]*p[j].
  cost: time O(n^3); space O(n^2)
  pitfall: Iterating by indices instead of by increasing interval length.
- subset_sum | subset sum, partition problem, boolean DP
  idea: Track which sums are reachable using a boolean table.
  formal: dp[w] becomes true if dp[w - a[i]] was true before item i was processed.
  cost: time O(nS); space O(S)
  pitfall: Calling it polynomial; O(nS) is pseudo-polynomial in the value of S.
- grid_paths | unique paths, grid DP, path counting with obstacles
  idea: Each cell is reachable from above or from the left.
  formal: dp[i][j] = dp[i-1][j] + dp[i][j-1], with obstacles forcing dp[i][j] = 0.
  cost: time O(mn); space O(n) with a rolling row
  pitfall: Initialising the first row and column to 1 without checking for obstacles.
- dp_state_design | DP state design, complexity of DP, defining states
  idea: The hard part of DP is choosing a state that is small enough yet captures everything the future needs.
  formal: A valid state makes the transition depend only on the state, never on the path taken to reach it.
  cost: runtime = states * transitions, which is the first thing to estimate
  pitfall: Adding an index to the state that carries no information, blowing up the table for nothing.

## Greedy and Backtracking
- greedy_choice | greedy choice property, exchange argument, when greedy works
  idea: Greedy works when a locally best choice is provably part of some optimal solution.
  formal: The exchange argument swaps an optimal solution's first choice for the greedy one without loss.
  cost: usually O(n log n), dominated by the sort that defines the greedy order
  pitfall: Assuming greedy works because it passes the sample tests, without an exchange argument.
- activity_selection | activity selection, interval scheduling, earliest finish
  idea: Always take the compatible activity that finishes earliest.
  formal: Sorting by finish time and greedily accepting non-overlapping activities is optimal by an exchange argument.
  cost: time O(n log n); space O(1)
  pitfall: Sorting by start time or by duration, both of which are provably suboptimal.
- fractional_knapsack | fractional knapsack, value per weight, greedy knapsack
  idea: Take items in decreasing value-per-weight order and split the last one.
  formal: Fractions make the greedy ratio order optimal, which is exactly why 0/1 knapsack needs DP instead.
  cost: time O(n log n)
  pitfall: Applying the ratio rule to the 0/1 version, where it can be arbitrarily bad.
- huffman | Huffman coding, prefix code, optimal encoding
  idea: Repeatedly merge the two least frequent symbols.
  formal: The merged tree gives a prefix-free code of minimum expected length sum(f_i * depth_i).
  cost: time O(n log n) with a priority queue
  pitfall: Merging the two most frequent symbols, which inverts the optimum.
- coin_change_greedy | greedy coin change failure, canonical coin systems
  idea: Taking the largest coin first is optimal only for canonical coin systems.
  formal: With coins {1, 3, 4} and target 6, greedy gives 4+1+1 while the optimum is 3+3.
  cost: greedy O(n); the DP that always works is O(nS)
  pitfall: Generalising from the Euro or Rupee system, which happens to be canonical.
- interval_merging | merge intervals, overlapping intervals, sweep
  idea: Sort by start and extend the current interval while the next one overlaps.
  formal: Intervals overlap when next.start <= current.end; otherwise the current interval is finalised.
  cost: time O(n log n); space O(n) for the output
  pitfall: Deciding overlap with a strict inequality when touching intervals should merge.
- backtracking_template | backtracking, choose explore unchoose, search tree
  idea: Make a choice, recurse, then undo the choice exactly.
  formal: The recursion explores a tree of partial solutions; correctness depends on the undo restoring the exact prior state.
  cost: exponential in the worst case, controlled by pruning
  pitfall: Forgetting the undo step, so state leaks into sibling branches.
- n_queens | N queens, queen placement, diagonal conflicts
  idea: Place one queen per row and reject conflicting columns and diagonals.
  formal: Diagonals are identified by row + col and row - col, so conflict checks are O(1) with three boolean arrays.
  cost: exponential but heavily pruned; practical up to about n = 14
  pitfall: Re-scanning the board for conflicts instead of maintaining the diagonal sets.
- sudoku_solver | sudoku, constraint propagation, cell candidates
  idea: Fill the most constrained empty cell first and backtrack on contradiction.
  formal: Choosing the cell with the fewest candidates shrinks the search tree dramatically.
  cost: exponential worst case; fast in practice with propagation
  pitfall: Scanning cells in fixed order, which explores an enormous tree.
- subsets_permutations | subsets, power set, permutations, combinations
  idea: Include or exclude each element for subsets; swap and recurse for permutations.
  formal: There are 2^n subsets and n! permutations, which bounds any enumeration.
  cost: subsets O(n * 2^n); permutations O(n * n!)
  pitfall: Appending the shared working list without copying, so all results alias one object.
- pruning | pruning, branch and bound, feasibility check
  idea: Abandon a branch as soon as it cannot beat the best solution found so far.
  formal: A bound function must never overestimate the achievable quality, or optimal solutions get cut.
  cost: same worst case, often orders of magnitude faster in practice
  pitfall: Using an unsound bound, which silently discards the optimum.
- greedy_vs_dp | greedy versus dynamic programming, choosing an approach
  idea: Greedy commits immediately; DP keeps every promising option.
  formal: If a locally optimal choice can be proven safe, greedy is O(n log n); otherwise DP explores the state space.
  cost: greedy typically O(n log n); DP typically polynomial in states
  pitfall: Reaching for DP on problems with a clean exchange argument, and for greedy on problems without one.

## Complexity and Notation
- big_o | Big O, upper bound, asymptotic notation
  idea: Big-O describes how the cost grows, ignoring constants and small inputs.
  formal: f(n) = O(g(n)) when there exist c > 0 and n0 such that f(n) <= c*g(n) for all n >= n0.
  cost: O(1) < O(log n) < O(n) < O(n log n) < O(n^2) < O(2^n) < O(n!)
  pitfall: Reading O as a tight bound; O(n) is technically true for a constant-time algorithm.
- omega_theta | Omega, Theta, tight bound, lower bound notation
  idea: Omega is a lower bound and Theta means the upper and lower bounds match.
  formal: f = Theta(g) exactly when f = O(g) and f = Omega(g).
  cost: use Theta when the analysis is tight, O when only an upper bound is proven
  pitfall: Writing O when the intent is Theta, then arguing about optimality.
- case_analysis | best case, average case, worst case
  idea: The same algorithm can have very different costs depending on the input.
  formal: Quicksort is O(n^2) worst case but Theta(n log n) expected over random pivots.
  cost: worst case bounds guarantees; average case predicts typical behaviour
  pitfall: Quoting the average case as if it were a guarantee in a latency-sensitive system.
- amortized_analysis | amortized analysis, aggregate method, potential method
  idea: Spread the cost of rare expensive operations over the many cheap ones.
  formal: The aggregate method bounds a sequence of n operations by T(n) and reports T(n)/n per operation.
  cost: dynamic array append is O(1) amortized despite O(n) resizes
  pitfall: Confusing amortized with average case; amortized holds for every sequence, not just random ones.
- recurrences | recurrence relation, master theorem, divide and conquer analysis
  idea: Divide-and-conquer costs are captured by a recurrence and solved with the master theorem.
  formal: For T(n) = aT(n/b) + f(n), compare f(n) with n^(log_b a) to pick one of the three cases.
  cost: merge sort T(n) = 2T(n/2) + O(n) resolves to O(n log n)
  pitfall: Applying the master theorem when f(n) is not polynomially comparable, as in T(n) = 2T(n/2) + n log n.
- space_complexity | space complexity, auxiliary space, memory analysis
  idea: Space counts the extra memory used, including the recursion stack.
  formal: Merge sort needs O(n) auxiliary space; quicksort needs O(log n) for its recursion.
  cost: report auxiliary space separately from input space
  pitfall: Calling a recursive algorithm O(1) space while its call stack is O(n) deep.
- recursion_tree | recursion tree, level sums, cost visualisation
  idea: Draw the recursion, sum each level, then sum the levels.
  formal: With cost n per level and log n levels, the total is n log n; a geometric decay makes the root dominate.
  cost: an intuition tool that also verifies the master theorem's answer
  pitfall: Assuming every level costs the same without checking the branching factor.
- ds_complexity_table | complexity cheat sheet, data structure operations, comparison table
  idea: Choosing a data structure is mostly a matter of matching operation costs to the access pattern.
  formal: Array indexes in O(1) but inserts in O(n); hash maps look up in expected O(1) without order; balanced BSTs are O(log n) with order.
  cost: memorise the table, but always ask which operations dominate the workload
  pitfall: Optimising an operation that runs rarely while ignoring the one in the hot loop.
"""


def _parse(raw: str) -> Dict[str, List[dict]]:
    """Parse the DSL above into {topic: [concept_dict, ...]}."""
    topics: Dict[str, List[dict]] = {}
    current_topic: str | None = None
    current: dict | None = None

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            current_topic = stripped[3:].strip()
            topics[current_topic] = []
            current = None
        elif stripped.startswith("- "):
            key, _, aliases = stripped[2:].partition("|")
            current = {
                "key": key.strip(),
                "name": key.strip().replace("_", " ").title(),
                "aliases": [a.strip() for a in aliases.split(",") if a.strip()],
                "topic": current_topic,
            }
            topics[current_topic].append(current)
        elif ":" in stripped and current is not None:
            field, _, value = stripped.partition(":")
            field = field.strip()
            if field in {"idea", "formal", "cost", "pitfall"}:
                current[field] = value.strip()
    return topics


CONCEPTS: Dict[str, List[dict]] = _parse(RAW)

#: Canonical display name of each concept's primary alias (used to build queries).
ALL_CONCEPTS: List[dict] = [c for concepts in CONCEPTS.values() for c in concepts]


def topic_names() -> List[str]:
    return list(CONCEPTS.keys())


def concepts_for(topic: str) -> List[dict]:
    return CONCEPTS[topic]


if __name__ == "__main__":  # quick sanity check: python -m src.data.dsa_content
    total = 0
    for topic, concepts in CONCEPTS.items():
        missing = [c["key"] for c in concepts
                   if not all(f in c for f in ("idea", "formal", "cost", "pitfall"))]
        print(f"{topic:35s} {len(concepts):3d} concepts  missing_fields={missing}")
        total += len(concepts)
    print(f"{'TOTAL':35s} {total:3d} concepts")
