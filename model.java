import java.util.*;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
class vector{
    double a;
    double b;
    vector(double a, double b){
        this.a = a;
        this.b = b;
    }
}
class operations{
    vector w;
    vector i;
    static double x;
    operations(vector w, vector i){
        this.w = w;
        this.i = i;
    }
    static double dot(vector w, vector i){
        return (w.a * i.a) + (w.b * i.b);
    }
    static int check(double p){
	    if(p >= 0.5){
	        return 1;
	    }else if(p < 0.5){
	        return 0;
	    }else{
            return 2;
        }
    }
    static double mse(double p, double t){
        x = p - t;
        return Math.pow(p-t,2);
    }
    static double derivative(double x){
        return 2*x;
    } 
    static vector vectorMultiply(vector v, double n){
        v.a *= n;
        v.b *= n;
        return v;
    } 
    static vector vectorSubstraction(vector v, vector n){
        v.a -= n.a;
        v.b -= n.b;
        return v;
    }
    static int randomIndex(int n, int x){
        return ThreadLocalRandom.current().nextInt(n,x);
    } 
}
class network{
    static double layer1, layer2;
    static Random rd = new Random();
    double learningRate;
    vector derror_dweight;
    double derror_dbias;
    vector weight;
    double bias;
    network(double learningRate){
        this.learningRate = learningRate;
    }
    public void generateWeightAndBias(){
        this.weight = new vector(rd.nextDouble(), rd.nextDouble());
        this.bias = rd.nextDouble();
    }
    public double sigmoid(double x){
	    double euler = Math.pow(2.7183, -x);
	    return 1 / (1 + euler);
    }
    public double sigmoidDerivative(double x){
        return sigmoid(x)*(1-sigmoid(x));
    }
    public double makePrediction(vector v){
        double dot = operations.dot(weight, v);
	    layer1 = dot + this.bias;
	    layer2 = sigmoid(layer1);
	    return layer2;
    }
    public void computeGradient(vector v, int target){
        double prediction = makePrediction(v);
        double derror_dprediction = operations.derivative(prediction - target);
        double dprediction_dlayer1 = sigmoidDerivative(layer1);
        vector dlayer1_dweight = v;
        double dlayer1_dbias = 1;

        derror_dbias = derror_dprediction * dprediction_dlayer1 * dlayer1_dbias;
        derror_dweight = operations.vectorMultiply(dlayer1_dweight, derror_dprediction * dprediction_dlayer1);  
    }
    public void updateParameters(double derror_dbias, vector derror_dweight){
        this.derror_dbias = derror_dbias;
        this.derror_dweight = derror_dweight;
        bias -= derror_dbias * learningRate;
        weight = operations.vectorSubstraction(weight, operations.vectorMultiply(derror_dweight,learningRate));
    }
    public ArrayList<Double> train(vector[] inputVectors, int[] targets,int iterations){
        ArrayList<Double> cumulativeErrors = new ArrayList<>();
        for(int i = 0; i < iterations; i++){
            int dataIndex = operations.randomIndex(0,inputVectors.length);
            vector inputVector = inputVectors[dataIndex];
            int target = targets[dataIndex];
            computeGradient(inputVector, target);
            updateParameters(derror_dbias,derror_dweight);
            double cumulativeError = 0;
            if (i % 100 == 0){
                for( dataIndex = 0; dataIndex < inputVectors.length; dataIndex++){
                    vector tempVector = inputVectors[dataIndex];
                    int tempTarget = targets[dataIndex];
                    double prediction = makePrediction(tempVector);
                    double error = Math.pow(prediction-tempTarget, 2);
                    cumulativeError += error;
                }
                cumulativeErrors.add(cumulativeError);
            }
        }
        return cumulativeErrors;
    }
}
public class model{
    public static void main(String[] args){
        vector v1 = new vector(3, 1.5);
        vector v2 = new vector(2, 1);
        vector v3 = new vector(4,1.5);
        vector v4 = new vector(3,4);
        vector v5 = new vector(3.5,0.5);
        vector v6 = new vector(2,0.5);
        vector v7 = new vector(5.5,1);
        vector v8 = new vector(1,1);
        vector[] inputVectors = {
            v1, v2, v3, v4, v5, v6, v7, v8
        };
        int[] targets = {0,1,0,1,0,1,1,0};
        double learningRate = 0.1;
        network myNetwork = new network(learningRate);
        myNetwork.generateWeightAndBias();
        ArrayList<Double> trainingError = myNetwork.train(inputVectors, targets, 10000);
        for(int i=0; i<trainingError.size(); i++){
            System.out.println(trainingError.get(i) + " ");
        }
    }
}
