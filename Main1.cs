using System;
using System.IO;

namespace NS_ANN
{
    public class Executor
    {
        static void Main(string[] args)
        {
            string[][] Dataset;
            double[][] stdDataset;
            double[] data;
            string[] colTypes;

            Console.WriteLine("Ciao e benvenuto nella Learning Part di Jarvis 0.1.");
            Console.ReadKey();

            //Segmento di prova per la standardizzazione dei dati in ingresso (Solo numerici e categorici, no audio/immagini)
            string buffer;
            string datasetPath = "C:/Users/Shea/Documents/Visual Studio 2015/Projects/ConsoleApplication1/Dataset.txt";

            Console.WriteLine("Ora leggo il Dataset " + datasetPath + ":");
            Console.ReadKey();

            int numSamples;
            int i = 0;

            StreamReader reader = new StreamReader(datasetPath);
            colTypes = reader.ReadLine().Split(' ');
            numSamples = File.ReadAllLines(datasetPath).Length - 1;
            Dataset = new string[numSamples][];
            data = new double[colTypes.Length];

            while ((buffer = reader.ReadLine()) != null)
            {
                Dataset[i++] = buffer.Split(',');
            };
            reader.Close();

            /*Console.WriteLine("\nDataset letto: \n");
            for (i = 0; i < Dataset.Length; i++)
            {
                Console.Write("\n");
                for (int j = 0; j < Dataset[0].Length; j++)
                    Console.WriteLine(Dataset[i][j]);
            }
            Console.ReadKey();*/
            Console.WriteLine("\nSuccesso. Standardizzo il dataset...");
            Console.ReadKey();

            Standardizer std = new Standardizer(Dataset, colTypes);
            stdDataset = std.StandardizeAll(Dataset);
            /*Console.WriteLine("Dati standardizzati:\n");
            helper.ShowMatrix(stdDataset, numSamples, stdDataset[0].Length);
            Console.ReadKey();*/
            //Fine segmento di Prova

            Console.WriteLine("Successo. Inizializzo la Rete Neurale...");
            Console.ReadKey();

            FFANN ffann = new FFANN(colTypes.Length, stdDataset[0].Length - Dataset[0].Length + 1); //Setting Network parameters & Creating it.

            Console.WriteLine("\nStruttura della Rete Neurale: \n");
            Helpers.ShowWeights(ffann);
            Console.ReadKey();

            Console.WriteLine("Successo, ora avvìo l'allenamento con Backpropagation...");
            Console.ReadKey();

            RPropPlus trainer = new RPropPlus(ffann, stdDataset);

            Console.WriteLine("\nNuova struttura della Rete Neurale: \n");
            Helpers.ShowWeights(ffann);
            Console.ReadKey();

            Console.WriteLine("Successo, inserisci dati di prova per effetuare una predizione: ");
            string[] buff;
            do
            {
                do
                {
                    buffer = Console.ReadLine();
                    buff = buffer.Split(',');
                } while (buff.Length != colTypes.Length - 1);

                data = std.GetStandardRow(buff);
                ffann.PredictShow(data);
                Console.ReadKey();
                Console.WriteLine("\nProvare ancora? 'y' = yes, others = no: ");
            } while (Console.ReadLine().Equals("y"));
        }
    }
}

