import numpy as np
from random import randrange
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
style.use("ggplot")
def create_data(l,variance=10,corr="pos",steps=2):
    X=[x for x in range(l)]
    y=[]
    val=1
    for i in range(l):
        value=val+randrange(-variance,stop=variance)
        y.append(value)
        if(corr=="pos"):
            val+=steps
        else:
            val-=steps
    return np.array(X,dtype=np.float64),np.array(y,dtype=np.float64)

def linear_regress(X,y):
    slope=(((np.mean(X)*np.mean(y))-(np.mean(X*y)))/((np.mean(X)**2)-np.mean(X**2)))
    coeff=np.mean(y)-slope*(np.mean(X))
    return slope,coeff

def main():

    X,y=create_data(100,variance=10,corr="pos")
    slope,coeff=linear_regress(X,y)
    linear_line=[]

    linear_line=np.array(linear_line)
    fig,(ax1,ax2,ax)=plt.subplots(3,1,sharex=True)
    # fig2,ax=plt.subplots(1,1)
    ax.scatter(X,y)
    ax.plot(X,linear_line,color="b")
    ax1.scatter(X,y)
    ax1.set_title("original data")

    ax2.plot(X,linear_line,color="g")
    ax2.set_title("the line")
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()
