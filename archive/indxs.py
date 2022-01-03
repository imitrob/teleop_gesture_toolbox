## Helper functions
def indx(ss, str):
    ''' Returns index of scene specified by name as parameter 'str'
    '''
    N = None
    for n,i in enumerate(ss):
        if i.NAME == str:
            N = n
    assert isinstance(N, int), "Scene not found"
    return N
