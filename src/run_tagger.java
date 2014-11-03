import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;
import java.util.List;
import java.util.Stack;


public class run_tagger {
	static String sentsTest, sentsOut, modelFile;
	
	static List<String> pennTreebankTags = Arrays.asList(
			"CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", 
			"NN", "NNS", "NNP", "NNPS", "PDT", "POS","PRP", "PRP$", "RB", "RBR", 
			"RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", 
			"VBZ", "WDT", "WP", "WP$", "WRB", "$", "#", "``", "''", "-LRB-", 
			"-RRB-", ",", ".", ":");
	
	static List<String> pennTreebankTagsWithStartAndEnd = Arrays.asList(
			"CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", 
			"NN", "NNS", "NNP", "NNPS", "PDT", "POS","PRP", "PRP$", "RB", "RBR", 
			"RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", 
			"VBZ", "WDT", "WP", "WP$", "WRB", "$", "#", "``", "''", "-LRB-", 
			"-RRB-", ",", ".", ":", "<s>", "</s>");
	
	static Hashtable<String, Double> transitionalProbabilities = new Hashtable<String, Double>();
	
	static double[][] viterbiMatrix;
	static String[][] backPointerMatrix;

	public static void main(String[] args) {
		if(args.length < 3) {
			System.out.println("Error in arguments");
		}
		
		sentsTest = args[0];
		modelFile = args[1];
		sentsOut = args[2];

		readModelFile(modelFile);
		List<String> taggedSentences = performTagging(sentsTest);
		printOutputFile(taggedSentences);
		
		
		if(args.length == 4) { //Perform 10-fold cross validation
			performTenFoldCrossValidation();
		}
	}
	
	/**
	 * ----------------------------------------------------------------------------
	 * METHODS FOR PERFORMING VITERBI ALGORITHM
	 */
	@SuppressWarnings("resource")
	static void readModelFile(String filename) {
		try { //Reads in the model file generated from build_tagger
			FileInputStream fis = new FileInputStream(filename);
			DataInputStream dis = new DataInputStream(fis);
			BufferedReader br = new BufferedReader(new InputStreamReader(dis));
			
			String sentence;

			while ((sentence = br.readLine()) != null) {
				String[] tokens = sentence.split("~~~");
				String word = tokens[0] + "~~~" + tokens[1];
				double probability = Double.valueOf(tokens[2]);
				transitionalProbabilities.put(word, probability);
			}
		} catch (Exception e) {
			
		}
	}
	
	@SuppressWarnings("resource")
	static List<String> performTagging(String filename) {
		List<String> taggedSentences = new ArrayList<String>();
		try { //Tag sentences using Viterbi Algorithm
			FileInputStream fis = new FileInputStream(filename);
			DataInputStream dis = new DataInputStream(fis);
			BufferedReader br = new BufferedReader(new InputStreamReader(dis));
			
			String sentence;
			
			while ((sentence = br.readLine()) != null) {
				
				String[] tokens = sentence.split(" ");
				
				Stack<String> output = viterbi(tokens);
				
				String taggedSentence = tokens[0] + "/" + output.pop();
				for(int i = 1; i < tokens.length; i++) {
					taggedSentence += " " + tokens[i] + "/" + output.pop();
					
				}
				taggedSentences.add(taggedSentence);
			}
			
		} catch (Exception e) {
			
		}
		return taggedSentences;
	}
	
	static void printOutputFile(List<String> taggedSentences) {
		try {
			File file = new File(sentsOut);
			 
			if (!file.exists()) {
				file.createNewFile();
			}
	
			FileWriter fw = new FileWriter(file.getAbsoluteFile());
			BufferedWriter bw = new BufferedWriter(fw);
			
			for(String sentence : taggedSentences) {
				bw.write(sentence + "\n");
			} 
	
			bw.close();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	static void setUpViterbiMatrix(int numOfWords) { //Create a new Viterbi matrix for a new sentence
		viterbiMatrix = new double[pennTreebankTags.size()][numOfWords];
		backPointerMatrix  = new String[pennTreebankTags.size()][numOfWords];
	}
	
	static Stack<String> viterbi(String[] observation) {
		
		int T = observation.length;
		int N = pennTreebankTags.size();
		setUpViterbiMatrix(T);
		
		double m = -1000000;

		for(int i = 0; i < N; i++) { //Initial setup, <s> | tag1
			String pos = pennTreebankTags.get(i) + "~~~<s>";
			
			viterbiMatrix[i][0] = getPosProbability(pos) * getWordProbability(pennTreebankTags.get(i), observation[0]);
			backPointerMatrix[i][0] = "0";
			if(viterbiMatrix[i][0]> m) {
				m = viterbiMatrix[i][0];

			}
		}
		//Calculate optimal path
		for(int i = 1; i < T; i++) {
			for(int j = 0; j < N; j++) {
				updateMatrix(observation[i], pennTreebankTags.get(j), i, j);
			}
		}
		
		//Termination, tag N | </s>
		double max = -1000000;
		int back = 0;
		for(int i = 0; i < N; i++) {
			String pos = "</s>~~~" + pennTreebankTags.get(i);
	
			double probability = getPosProbability(pos) * viterbiMatrix[i][T-1];
			if(probability > max) {
				max = probability;
				back = i;
			}
		}

		return performBackTracking(back, T);
	}
	
	static Stack<String> performBackTracking(int last, int T) {
		Stack<String> backTracking = new Stack<String>(); 
		
		String backpointer = backPointerMatrix[last][T-1];
		backTracking.push(pennTreebankTags.get(last));
		
		for(int i = T-2; i >= 0; i--) { //Gets the optimal tags for the sentence
			
			String[] tokens = backpointer.split(",");
			int row = Integer.parseInt(tokens[0]);
			
			backpointer = backPointerMatrix[row][i];
			backTracking.push(pennTreebankTags.get(row));
		}
		return backTracking;
	}
	
	static void updateMatrix(String currentWord, String currentTag, int wordPosition, int tagPosition) {
		double max = -1000000;
		String back = "";
		double wordProbability = getWordProbability(currentTag, currentWord);
		for(int i = 0; i < pennTreebankTags.size(); i++) {	//Does the actual calculation for each word and tag
			
			String pos = pennTreebankTags.get(i) + "~~~" + currentTag;
			
			double posProbability = getPosProbability(pos);
			double probability = viterbiMatrix[i][wordPosition-1] * wordProbability * posProbability;
			if(probability > max) {
				max = probability;
				back = String.valueOf(i) + "," + String.valueOf(wordPosition);
			}
		}	
		
		viterbiMatrix[tagPosition][wordPosition] = max;
		backPointerMatrix[tagPosition][wordPosition] = back;
	}
	
	static double getWordProbability(String pos, String word) {
		String combine = pos + "~~~" + word;

		if(transitionalProbabilities.containsKey(combine)) {
			return transitionalProbabilities.get(combine);
		} else {
			return transitionalProbabilities.get("<unknown>~~~word");
		}
	}
	
	static double getPosProbability(String pos) {
		if(transitionalProbabilities.containsKey(pos)) {
			return transitionalProbabilities.get(pos);
		} else {
			return transitionalProbabilities.get("<unknown>~~~pos");	
		}
	}
	/**
	 * METHODS FOR PERFORMING VITERBI ALGORITHM
	 * ----------------------------------------------------------------------------
	 */
	
	/**
	 * ----------------------------------------------------------------------------
	 * METHODS FOR PERFORMING 10-FOLD CROSS VALIDATION
	 */
	static void performTenFoldCrossValidation() {
		float[] totalCount = new float[pennTreebankTags.size()];
		float[] correctCount = new float[pennTreebankTags.size()];
		float[][] incorrectCount = new float[pennTreebankTags.size()][pennTreebankTags.size()];
		
		for(int i = 0; i < 10; i++) {
			transitionalProbabilities = new Hashtable<String, Double>();

			readModelFile("Tenfoldmodel" + i);
			List<String> testSentences = extractSentencesFromFile("Tenfoldtest"+i);
			List<String> taggedSentences = new ArrayList<String>();
			
			for(String sentence : testSentences) {
				
				String[] tokens = sentence.split(" ");
				
				Stack<String> output = viterbi(tokens);
				
				String taggedSentence = tokens[0] + "/" + output.pop();
				for(int j = 1; j < tokens.length; j++) {
					taggedSentence += " " + tokens[j] + "/" + output.pop();
					
				}
				taggedSentences.add(taggedSentence);
			}
			List<String> expectedAnswers =  extractSentencesFromFile("TenfoldAnswer"+i);		
			
			for(int j = 0; j < expectedAnswers.size(); j++) {
				
				
				String output = taggedSentences.get(j);
				String expected = expectedAnswers.get(j);
				
				String[] outputTokens = output.split(" ");
				String[] expectedTokens = expected.split(" ");
				
				for(int k = 0; k < outputTokens.length; k++) {
					String[] expectedPosTokens = expectedTokens[k].split("/");
					String pos = expectedPosTokens[expectedPosTokens.length-1];
					totalCount[pennTreebankTags.indexOf(pos)] += 1;

					if(outputTokens[k].equals(expectedTokens[k])) {
						String[] outputPosTokens = outputTokens[k].split("/");
						pos = outputPosTokens[outputPosTokens.length-1];
						correctCount[pennTreebankTags.indexOf(pos)] += 1;
					} else {
						int correctPos = pennTreebankTags.indexOf(pos);
						String[] outputPosTokens = outputTokens[k].split("/");
						pos = outputPosTokens[outputPosTokens.length-1];
						incorrectCount[correctPos][pennTreebankTags.indexOf(pos)] += 1;
					}
				}
				
			}
			float correct = 0;
			float total = 0;
			for(int z = 0; z < correctCount.length; z++) {
				correct += correctCount[z];
				total += totalCount[z];
			}
			System.out.println("Run " + i + "score: " + correct/total);
		}
		
		for(int j = 0; j < pennTreebankTags.size(); j++) {
			System.out.println("Accuracy for POS tag: " + pennTreebankTags.get(j) + " is: " + correctCount[j]/totalCount[j]);
			float mostWrong = 0;
			String mostWrongTag = "";
			for(int k = 0; k < pennTreebankTags.size(); k++) {
				if(incorrectCount[j][k] > mostWrong) {
					mostWrong = incorrectCount[j][k];
					mostWrongTag = pennTreebankTags.get(k);
				}
			}
			System.out.println("Accuracy for POS tag: " + pennTreebankTags.get(j) + " is: " + correctCount[j]/totalCount[j] +
					" Most wrong: " + mostWrongTag + " at: " + mostWrong/totalCount[j]);
		}
	}
	
	@SuppressWarnings("resource")
	static List<String> extractSentencesFromFile(String filename) {
		List<String> sentences = new ArrayList<String>();
		
		try {
			FileInputStream fis = new FileInputStream(filename);
			DataInputStream dis = new DataInputStream(fis);
			BufferedReader br = new BufferedReader(new InputStreamReader(dis));
			
			String sentence;

			while ((sentence = br.readLine()) != null) {
				sentences.add(sentence);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return sentences;
	}
	/**
	 * METHODS FOR PERFORMING 10-FOLD CROSS VALIDATION
	 * ----------------------------------------------------------------------------
	 */
}
