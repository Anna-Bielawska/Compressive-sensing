'''
   Author: Anna Bielawska
   Field of study: Mathematics and Statistics
   Dissertation: Compressive sensing - the new sampling method.
   Institution of studies: Wrocław University of Science and Technology
   Year of thesis defence: 2021
   
   The code written below was used for computer simulations included in the diploma thesis.
   It contains 4 original part-implementations of optimization algorithms: Basis Pursuit, Basic Thresholding, 
   Iterative Hard Thresholding, Hard Thresholding Pursuit, with use of scipy library. 
   There already existed an implementation of the 5th algorithm of interest, OMP - Orthogonal Matching Pursuit.
'''

################################################ BASIS PURSUIT ##################################################

def reconstructBP(vec, Mat, tolerance = 1e-16):
    
    '''Basis pursuit algorithm as in the thesis,
       recovers a nearly-sparse vector res from vec and Mat;
       Mat - sampling matrix,
       vec - vector of measurements,
       tolerance - magnitude of error to be tolerated'''
    
    # A function to be minimized, min ||x||_1
    fun = lambda x: sum(abs(x))
           
    # Length of the original signal == number of columns in the matrix Mat
    Z = np.shape(Mat)[1]
    
    # Constraints of the problem
    cons = {'type': 'eq', 'fun': lambda x: Mat.dot(x) - vec}
    
    # Find solution to the optimization problem
    res = scipy.optimize.minimize(fun, x0 = np.ones(Z), method='SLSQP', constraints=cons, tol = tolerance)['x']
    
    return res


    
############################################### BT - Basic Thresholding ############################################

def Ls(Arr, s, N):
    
    '''A function returning a list of s non-zero largest (in terms of module) indices;
       Arr - an array to be filtered,
       s - a number of non-zero coefficients,
       N - lenght of signal to be reconstructed
    '''
    
    if s > N:
        s = N
    Arr_1 = np.abs(Arr)
    Arr_sorted = sorted(Arr_1, reverse = True)
    
    # Find the boundary coefficient
    threshold = Arr_sorted[s-1]
    
    # Find indices of coefficients to be rid of (zeroed)
    Ls_index_0 = np.where(Arr_1 < threshold)
    
    # Change into a list of indices
    Ls_index_0 = np.concatenate(Ls_index_0).tolist()
    
    return Ls_index_0


def reconstructBT(A, y, s, tolerance = 1e-5):
    
    '''Basic thresholding algorithm as in the thesis,
       recovers a sparse vector res.x from y and A;
       A - sampling matrix,
       y - vector of measurements,
       s - number of non-zero coefficients in the signal to be reconstructed,
       tol = tolerance (error)'''
    
    # A function to be minimized (||Ax - y||_2)^2
    fun = lambda x: sum((A.dot(x) - y)**2)
    
    # Length of original signal
    N = np.shape(A)[1]
    
    # Dictionary for the estimator
    D = Ls(np.transpose(A).dot(y), s, N)

    # Constraints of the problem, x[D] = 0, indices of x that are in D must be equal to 0
    cons = {'type': 'eq', 'fun': lambda x: x[D]}
    
    # Find solution to the minimization problem
    res = scipy.optimize.minimize(fun, x0 = np.zeros(N), method='SLSQP', constraints=cons, tol = tolerance)    
    
    # Return a sparse vector x of the solution
    return res.x


############################################# IHT - Iterative Hard Thresholding ###################################

def hardThreshold(vec, size):
    
    '''A function returning a vector with small coefficients equal to zero;
    vec - vector to be filtered,
    size = number of non-zero coefficients to be left unchanged'''
    
    new_vec = sorted(abs(vec), reverse = True)
    thr = new_vec[size-1]
    
    #np.array of True/False
    j = abs(vec) < thr
    
    #where True, substitute with 0
    vec[j] = 0
    
    return vec


def IHT(A, y, s, Its=200, tol=0.0001):
    
    '''Iterative hard thresholding algorithm as in the thesis,
       recovers a sparse vector xhat from y and A;
       A - sampling matrix,
       y - vector of measurements,
       s - number of non-zero coefficients in the signal to be reconstructed,
       Its - number of maximum algorithm iterations,
       tol = tolerance (error)'''

    # Length of original signal
    Length = np.shape(A)[1]

    # Initial estimate
    xhat = np.zeros(Length)

    for t in range(Its):

        # Pre-threshold value
        gamma = xhat + np.dot(np.transpose(A), y-A.dot(xhat))

        # Estimate the signal (by hard thresholding)
        xhat = hardThreshold(gamma, size=s)

        # Stopping criteria
        if sum(abs(y-A.dot(xhat))) < tol:
            break

    return xhat


############################################ HTP - Hard Thresholding Pursuit ############################################

def HTP(A, y, s, Its=100, tol=0.0001):
    
    '''Hard thresholding pursuit algorithm as in the thesis,
       recovers a sparse vector xhat from y and A;
       A - sampling matrix,
       y - vector of measurements,
       s - number of non-zero coefficients in the signal to be reconstructed,
       Its - number of maximum algorithm iterations,
       tol = tolerance (error)'''

    # A function to be minimized
    fun = lambda x: sum((y - A.dot(x))**2)
    
    # Length of original signal
    N = np.shape(A)[1]

    # Initial estimate
    xhat = np.zeros(N)

    for t in range(Its):
        
        # Pre-threshold value
        gamma = xhat + np.dot(np.transpose(A), y-A.dot(xhat))

        # Find the dictionary in k-th step
        D_k = Ls(gamma, s, N)
        
        # Constraints for the problem
        cons = {'type': 'eq', 'fun': lambda x: x[D_k]}

        sol = scipy.optimize.minimize(fun, x0 = xhat, method='SLSQP', constraints=cons, tol = tol)    
        
        # Estimated solution
        xhat = sol.x

        # Stopping criteria
        if sum(abs(y-A.dot(xhat))) < tol:
            break

    return xhat
