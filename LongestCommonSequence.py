def Max(a,b):
    if a >= b:
        return a
    else:
        return b

def LongestCommonSequence(s1, s2):
    if (s1 == None):
        raise ValueError()
        
    if (s2 == None):
        raise ValueError()
        
    len1 = len(s1)
    len2 = len(s2)
    
    if (len1 == 0) or (len2 == 0):
        return (0, "")
    
    if s1[-1] == s2[-1]:
        res = LongestCommonSequence(s1[:-1], s2[:-1])
        
        return (1 + res[0], res[1] + s1[-1])
    else:
        res1 = LongestCommonSequence(s1[:-1], s2)
        res2 = LongestCommonSequence(s1, s2[:-1])
        
        if (res1[0] >= res2[0]):
            return res1
        else:
            return res2

# LongestCommonSequence("ABCDGH", "AEDFHR")

# LongestCommonSequence("AGGTAB", "GXTXAYB")