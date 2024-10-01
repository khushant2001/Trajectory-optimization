def nested_search(d, keyList):
    stripDict = d.copy()
    for k in keyList:
        stripDict = stripDict[k]
    return stripDict