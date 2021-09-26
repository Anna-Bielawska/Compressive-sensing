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