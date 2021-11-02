

def blendTwoSignals():
    ''' ARCHIVE FUNC
    '''

    position = np.array(settings.joints)
    velocity = np.array(settings.velocity)

    def sigmoid(x, k=3):
        if x == 0: return 0.
        if x == 1: return 1.
        return 1. / (1. + (x/(1-x))**(-k))
    ## smoothing
    maxblend = len(settings.md._goal.trajectory.points)/5
    for n in range(0,int(maxblend)):
        blend_ratio = sigmoid(float(n)/maxblend)
        blend_ratio_inv = sigmoid((maxblend-float(n)) / maxblend)

        pt1 = np.multiply(blend_ratio, np.array(settings.md._goal.trajectory.points[n].positions)) # float x 1darr -> 1darr
        pt2 = np.multiply(blend_ratio_inv, position) # float x 1darr -> 1darr

        vt1 = np.multiply(blend_ratio, np.array(settings.md._goal.trajectory.points[n].velocities))
        vt2 = np.multiply(blend_ratio_inv, velocity)
        settings.md._goal.trajectory.points[n].positions = np.add(pt1, pt2)
        settings.md._goal.trajectory.points[n].velocities = np.add(vt1, vt2)
        #print("n", n, "blend_ratio", blend_ratio, "blend_ratio_inv", blend_ratio_inv, "pt1",pt1, "pt2", pt2)
        #input()
    ##




# FILTER savitzky_golay
n = 100 # n elements
pts = self._goal.trajectory.points[-n:] # pick the last n elements
pts.extend(goal.trajectory.points[0:n]) # pick first n elements
for i in range(0,7):
    poss = [pt.positions[i] for pt in pts] # [2*n x 7]
    poss = self.savitzky_golay(poss, 77, 2, rate=10)

    vels = [pt.velocities[i] for pt in pts]
    vels = self.savitzky_golay(vels, 77, 2, rate=10)

    for j in range(0,n):
        self._goal.trajectory.points[-(n-j)].positions[i] = poss[j]
        goal.trajectory.points[j].positions[i] = poss[n+j]

        self._goal.trajectory.points[-(n-j)].velocities[i] = vels[j]
        goal.trajectory.points[j].velocities[i] = vels[n+j]
