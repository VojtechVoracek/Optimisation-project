from sklearn.datasets import load_diabetes
from plots import *

if __name__ == "__main__":
    dimension = 5
    max_epochs = 2000
    tol = 1e-7
    q = 0.2
    runs = 100
  

    #SPHERE ========================
    #plot_runs(SphereFunction(),max_epochs, runs, tol, True, q, subplots=np.array([2,3,5,10]))

    #COMPONENT FUNCTION ============
    #plot_runs(ComponentFunction() ,max_epochs, runs, tol, True, q)
    
    #NON CONVEX FUNCTION ===========
    plot_runs(Non_Convex_Function(10), max_epochs, runs, tol, True, q)

    #LINEAR REGRESSION =============
    # data = load_diabetes()
    # A = data['data']
    # b = np.expand_dims(data['target'], 1)
    # plot_runs(LinearRegression(A,b),max_epochs, runs, tol, True, q, fixed_plot_k=10000)
