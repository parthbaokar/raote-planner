import random
def RandFloats(Size):
    Scalar = 1.0
    VectorSize = Size
    RandomVector = [random.random() for i in range(VectorSize)]
    RandomVectorSum = sum(RandomVector)
    RandomVector = [Scalar*i/RandomVectorSum for i in RandomVector]
    return RandomVector

from numpy.random import multinomial
import math
def RandIntVec(ListSize, ListSumValue, Distribution='Normal'):
    """
    Inputs:
    ListSize = the size of the list to return
    ListSumValue = The sum of list values
    Distribution = can be 'uniform' for uniform distribution, 'normal' for a normal distribution ~ N(0,1) with +/- 5 sigma  (default), or a list of size 'ListSize' or 'ListSize - 1' for an empirical (arbitrary) distribution. Probabilities of each of the p different outcomes. These should sum to 1 (however, the last element is always assumed to account for the remaining probability, as long as sum(pvals[:-1]) <= 1).  
    Output:
    A list of random integers of length 'ListSize' whose sum is 'ListSumValue'.
    """
    if type(Distribution) == list:
        DistributionSize = len(Distribution)
        if ListSize == DistributionSize or (ListSize-1) == DistributionSize:
            Values = multinomial(ListSumValue,Distribution,size=1)
            OutputValue = Values[0]
    elif Distribution.lower() == 'uniform': #I do not recommend this!!!! I see that it is not as random (at least on my computer) as I had hoped
        UniformDistro = [1/ListSize for i in range(ListSize)]
        Values = multinomial(ListSumValue,UniformDistro,size=1)
        OutputValue = Values[0]
    elif Distribution.lower() == 'normal':
        """
        Normal Distribution Construction....It's very flexible and hideous
        Assume a +-3 sigma range.  Warning, this may or may not be a suitable range for your implementation!
        If one wishes to explore a different range, then changes the LowSigma and HighSigma values
        """
        LowSigma    = -3#-3 sigma
        HighSigma   = 3#+3 sigma
        StepSize    = 1/(float(ListSize) - 1)
        ZValues     = [(LowSigma * (1-i*StepSize) +(i*StepSize)*HighSigma) for i in range(int(ListSize))]
        #Construction parameters for N(Mean,Variance) - Default is N(0,1)
        Mean        = 0
        Var         = 1
        #NormalDistro= [self.NormalDistributionFunction(Mean, Var, x) for x in ZValues]
        NormalDistro= list()
        for i in range(len(ZValues)):
            if i==0:
                ERFCVAL = 0.5 * math.erfc(-ZValues[i]/math.sqrt(2))
                NormalDistro.append(ERFCVAL)
            elif i ==  len(ZValues) - 1:
                ERFCVAL = NormalDistro[0]
                NormalDistro.append(ERFCVAL)
            else:
                ERFCVAL1 = 0.5 * math.erfc(-ZValues[i]/math.sqrt(2))
                ERFCVAL2 = 0.5 * math.erfc(-ZValues[i-1]/math.sqrt(2))
                ERFCVAL = ERFCVAL1 - ERFCVAL2
                NormalDistro.append(ERFCVAL)  
            #print "Normal Distribution sum = %f"%sum(NormalDistro)
            Values = multinomial(ListSumValue,NormalDistro,size=1)
            OutputValue = Values[0]
        else:
            raise ValueError ('Cannot create desired vector')
        return OutputValue
    else:
        raise ValueError ('Cannot create desired vector')
    return OutputValue