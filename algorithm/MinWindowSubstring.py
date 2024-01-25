from collections import Counter
def MinWindowSubstring(strArr):

    N,K = strArr[0], strArr[1]
    K_c = Counter(K)
    window_size = len(K)
    for window_size in range(len(K),len(N)) :
        
        for i in range(0,len(N)-window_size+1):
            tmp = N[i:window_size+i]
            tmp_c = Counter(tmp)
        
            tmp_f = [True if K_c[key] <= tmp_c[key] else False for key in K_c ]
            if all(tmp_f) : 
                return tmp
    # code goes here
    return N

# keep this function call here 

# print(MinWindowSubstring(input()))
print(MinWindowSubstring(["aaffhkksemckelloe", "fhea"]))
# affhkkse

# print(MinWindowSubstring(["aaabaaddae", "aed"]))

# Input: ["ahffaksfajeeubsne", "jefaa"]
# Output: aksfaje

### 투 포인터로 풀기.