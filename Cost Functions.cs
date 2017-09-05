using System;

namespace NS_ANN
{
    //Questa interfaccia definisce la struttura di OGNI funzione di attivazione principale, la quale DEVE ereditare da Activation.
    public interface Activation
    {
        double Compute(double x); //Compute the Activation Function.
        double Dcompute(double x); //Compute de first derivative of the activation function.
        double Err(double real, double obtained); //Compute the Error Function for the Activation Function.
        double Derr1(double real, double obtained);//Compute the First Derivative of the Error Function for the Activation Function.

    }



    class LogisticSigmoid : Activation
    {
        public double Compute(double x)
        {
            return (1 / (1 + Math.Pow(Math.E, -x)));
        }

        public double Dcompute(double x)
        {
            return ((Compute(x) * (1 - Compute(x)) + 0.1)); /*anti-plateau*/
        }

        public double Err(double real, double obtained)
        {
            return (0.5 * Math.Pow((obtained - real), 2));
        }

        public double Derr1(double real, double obtained)
        {
            return ((obtained - real) * Dcompute(obtained));
        }
    }



    class HiperTan : Activation
    {
        public double Compute(double x)
        {
            return (2 / (1 + Math.Pow(Math.E, -2 * x)) - 1);
        }

        public double Dcompute(double x)
        {
            return (1 - Math.Pow(Compute(x), 2) + 0.1);
        }

        public double Err(double real, double obtained)
        {
            return (0.5 * Math.Pow((obtained - real), 2));
        }

        public double Derr1(double real, double obtained)
        {
            return ((obtained - real) * Dcompute(obtained));
        }
    }



    class HeivisideStep : Activation
    {
        public double Compute(double x)
        {
            if (x > 0)
                return 1;
            return 0;
        }

        public double Dcompute(double x)
        {
            return 0;
        }

        public double Err(double real, double obtained)
        {
            return (0.5 * Math.Pow((obtained - real), 2));
        }

        public double Derr1(double real, double obtained)
        {
            return (obtained - real) * Dcompute(obtained);
        }
    }



    class identity : Activation
    {
        public double Compute(double x)
        {
            return x;
        }

        public double Dcompute(double x)
        {
            return 1;
        }

        public double Err(double real, double obtained)
        {
            return (0.5 * Math.Pow((obtained - real), 2));
        }

        public double Derr1(double real, double obtained)
        {
            return ((obtained - real) * Dcompute(obtained));
        }
    }



    //ritorna il vettore probabilità
    class Softmax
    {
        public Softmax(Layer layer)
        {
            double TotalDivisor = 0;
            for (int i = 0; i < layer.perceptron.Length; i++)
                TotalDivisor += Math.Pow(Math.E, layer.perceptron[i].getAction());
            for (int i = 0; i < layer.perceptron.Length; i++)
                layer.perceptron[i].setAction(Math.Pow(Math.E, layer.perceptron[i].getAction()) / TotalDivisor);
        }
    }
}