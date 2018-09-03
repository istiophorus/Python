def LongestCommonSubstring(s1, s2):
    if (s1 == None):
        raise ValueError()
        
    if (s2 == None):
        raise ValueError()
        
    len1 = len(s1)
    len2 = len(s2)
        
    maxLength = 0
    result = None
        
    for startIx1 in range(len1):
        for startIx2 in range(len2):
            ix1 = startIx1
            ix2 = startIx2
            
            counter = 0
            
            while ix1 < len1 and ix2 < len2 and s1[ix1] == s2[ix2]:
                counter += 1
                if counter > maxLength:
                    maxLength = counter
                    result = s1[startIx1:startIx1 + counter]
                ix1 += 1
                ix2 += 1
               
    return (maxLength, result)

# LongestCommonSubstring("abcdxyz", "xyzabcd")

# LongestCommonSubstring("zxabcdezy", "yzabcdezx")

