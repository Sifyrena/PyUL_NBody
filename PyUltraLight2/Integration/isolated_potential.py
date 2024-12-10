###### LUNA'S ISOLATED POTENTIAL
# With FW Changes to Ensure Compatibility


####################### PADDED POTENTIAL FUNCTIONS

#safeInverse

def safeInverse3D(x):
    # Safe inverse for 3D arrays
    
    y = x*0
    for ind in np.ndindex(x.shape):
        if x[ind] > 0:
                    #x[i, j] = 1.0 / float(x[i, j])
                    y[ind] = np.reciprocal(x[ind])
    return y

# greenArray 

def greenArray(xyz):
    # make grid for Green's function
    x2, y2, z2 = np.meshgrid(
        xyz, xyz, xyz, 
        sparse=True, indexing='ij',
    )
    gplane = ne.evaluate("x2 + y2 + z2")
    return(gplane)

# makeDCTGreen

def makeDCTGreen(n):
    # make Green's function using a DCT
    x2 = np.arange(0, n + 1)**2
    arr = greenArray(x2)
    arr = np.sqrt(arr)
    arr = safeInverse3D(arr)
    arr[0,0,0] = 3*np.log(np.sqrt(3) + 2) - np.pi/2 #calculated in https://arxiv.org/pdf/chem-ph/9508002.pdf
    # no normalization below matches un-normalization in Mathematica
    import scipy.fft
    with scipy.fft.set_workers(num_threads):
        green =  scipy.fft.dctn(arr, type = 1, norm = None)
    return(green)
    
#makeEvenArray

def makeEvenArray(arr):
    # defined for a square 2D array
    # doubles the size of the array to make it even
    # this allows us to use the DCT
    nRow, nCol = np.shape(arr)
    # first do rows 
    arow = np.flip(arr[1:nRow-1, :], axis = 0)
    arr = np.concatenate((arr, arow))
    #now do columns
    acol = np.flip(arr[:, 1:nCol-1], axis = 1)
    arr = np.concatenate((arr, acol), axis = 1)
    #return
    return(arr)

#Plane Convolve2

def planeConvolve(green, plane, n, fft_plane, ifft_plane):
    # Convolve Green's function for each plane
    bigplane = np.pad(plane, ((0,n), (0,n)), 'constant', constant_values=0)
    green1 = makeEvenArray(green) # defined above
    temp = -1 * green1 * fft_plane(bigplane) 
    bigplane = ifft_plane(temp)
    bigplane = bigplane[:-n, :-n]
    return(bigplane)

#Isolated Potential2

def isolatedPotential(rho, green, l, fft_X, ifft_X, fft_plane, ifft_plane):
    # the main function
    # pads x-direction
    # then takes each plane, pads it, and convolves it with the appropriate Green's function plane
    n,n,n = np.shape(rho)
    rhopad = np.pad(rho, ((0,n), (0,0), (0,0)), 'constant', constant_values=0)
    #rhopad = scipy.fftpack.fftn(rhopad, axes = (0,)) #transform x-axis
    rhopad = fft_X(rhopad)
    # ndx is the index for green (so we don't have to waste memory storing EvenGreen)
    ndx = np.concatenate((np.arange(n + 1), np.flip(np.arange(1,n)) ))
    # the for loop
    for i in range(0,2*n):
        plane = rhopad[i, :, :]
        rhopad[i, :, :] = PC_jit(green[ndx[i], :, :], plane, n, fft_plane, ifft_plane)
    # - - - - - - - - - - - - - - - - - - - - - -    
    #rhopad = (1/n**2) * scipy.fftpack.ifftn(rhopad, axes = (0,)) #inverse transform x-axis
    rhopad = (1/n**2) * ifft_X(rhopad) # normalization
    rho = (l**2) * rhopad[:-n, :, :] #potential has units of length^2 in this system
    return(rho.real)


#FW Attempts to Speed This Up


def planeConvolveSP(green, plane, n, fft_plane, ifft_plane):
    
    # Convolve Green's function for each plane
    
    bigplane = fft_plane(np.pad(plane, ((0,n), (0,n)), 'constant', constant_values=0))
    
    # print(np.shape(green), "SHAPE OF GREEN")
    
    green1 = makeEvenArray(green) # defined above
    
    temp = -1*green1*bigplane

    return ifft_plane(temp)[:-n, :-n]


def isolatedPotentialSP(rho, green, l, fft_X, ifft_X, fft_plane, ifft_plane, resol, ndx):
    # the main function
    # pads x-direction
    # then takes each plane, pads it, and convolves it with the appropriate Green's function plane
    n = resol
    rhopad = np.pad(rho, ((0,n), (0,0), (0,0)), 'constant', constant_values=0)
    #rhopad = scipy.fftpack.fftn(rhopad, axes = (0,)) #transform x-axis
    rhopad = fft_X(rhopad)
    
    # loop
    for i in range(0,2*n):
        plane = rhopad[i, :, :]
        rhopad[i, :, :] = PC_jit(green[ndx[i], :, :], plane, n, fft_plane, ifft_plane)
    # - - - - - - - - - - - - - - - - - - - - - -    
    #rhopad = (1/n**2) * scipy.fftpack.ifftn(rhopad, axes = (0,)) #inverse transform x-axis
    rhopad = (1/n**2) * ifft_X(rhopad) # normalization
    rho = (l**2) * rhopad[:-n, :, :] #potential has units of length^2 in this system
    return (rho.real)

####################### CONVERTING PARAMS TO CODE UNITS