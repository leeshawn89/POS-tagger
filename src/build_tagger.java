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
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.List;


public class build_tagger {
	
	static String sentsTrain, sentsDev, modelFile;
	
	static List<String> pennTreebankTags = Arrays.asList(
			"CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", 
			"NN", "NNS", "NNP", "NNPS", "PDT", "POS","PRP", "PRP$", "RB", "RBR", 
			"RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", 
			"VBZ", "WDT", "WP", "WP$", "WRB", "$", "#", "``", "''", "-LRB-", 
			"-RRB-", ",", ".", ":", "<s>", "</s>");
	
	static List<String> pennTreebankTagsNoStart = Arrays.asList(
			"CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", 
			"NN", "NNS", "NNP", "NNPS", "PDT", "POS","PRP", "PRP$", "RB", "RBR", 
			"RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", 
			"VBZ", "WDT", "WP", "WP$", "WRB", "$", "#", "``", "''", "-LRB-", 
			"-RRB-", ",", ".", ":");
	
	static Hashtable<String, Integer> conditionalCount = new Hashtable<String, Integer>();
	static Hashtable<String, Integer> totalCount = new Hashtable<String, Integer>();
	
	static Hashtable<String, Double> transitionalProbabilitiesPos = new Hashtable<String, Double>();
	static Hashtable<String, Double> transitionalProbabilitiesWords = new Hashtable<String, Double>();
	
	static int[][] distinctPosTags = new int[pennTreebankTags.size()][pennTreebankTags.size()];
	static Hashtable<String, String> distinctWordDictionary = new Hashtable<String, String>();
	
	static int[] distinctPosCount = new int[pennTreebankTags.size()];
	static Hashtable<String, Integer> distinctWordCount = new Hashtable<String, Integer>();
	
	static Hashtable<String, Double> transitionalProbabilities = new Hashtable<String, Double>();
	
	public static void main(String[] args) {
		sentsTrain = args[0];
		sentsDev = args[1];
		modelFile = args[2];

		countPosAndWords(sentsTrain);
		addOneSmoothing();
		saveToModelFile();	
		 
		//Uncomment these lines to perform 10-fold cross validation
		/*
		List<String> sentences = extractSentencesFromFile(sentsTrain);
		ArrayList<ArrayList<String>> bins = tenfoldCrossvalidationSplit(sentences);
		performTenfoldCrossvalidation(bins);	
		*/
	}
	
	/**
	 * -------------------------------------------------------------------------------------
	 * GENERAL METHODS FOR COUNTING POS AND WORDS
	 */
	@SuppressWarnings("resource")
	static void countPosAndWords(String filename) {
		try {
			FileInputStream fis = new FileInputStream(filename);
			DataInputStream dis = new DataInputStream(fis);
			BufferedReader br = new BufferedReader(new InputStreamReader(dis));
			
			String sentence;
			String[] tokens;
			String previousPos = "";

			while ((sentence = br.readLine()) != null) {
				//Adds the start and end tag to the sentence as required by Viterbi Algorithm
				sentence = "<s> " + sentence + " </s>";
				tokens = sentence.split(" ");
				
				for(String token : tokens) {
					if(token.equals("<s>")) { //Counts the first word with the start tag e.g <s> | tag1
						previousPos = "<s>";
						updateTotalCount(token);
						
					} else if (token.equals("</s>")) { //Counts the last word with the start tag e.g tagN | </s>
						updateTotalCount(token);
						updateDistinctPosCount(token, previousPos);
						updateConditionalCount(token + "~~~" + previousPos);
					} else { //Counts word|tag and tag-1 | tag
						String[] posTokens = token.split("/");
						String word = "";
						String pos = "";
						
						pos = posTokens[posTokens.length-1];
						word = posTokens[0];
						for(int i = 1; i < posTokens.length-1; i++) {
							word += "/" + posTokens[i];
						}
						
						updateTotalCount(pos);
						updateTotalCount(word);
						updateDistinctWordCount(pos, word);
						updateConditionalCount(pos + "~~~" + word); 
						updateDistinctPosCount(pos, previousPos);
						updateConditionalCount(previousPos + "~~~" + pos);
						previousPos = pos;
					}
				}
			}
			finalizeDistinctPosCount();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	static void updateDistinctPosCount(String pos, String previousPos) {
		distinctPosTags[pennTreebankTags.indexOf(previousPos)][pennTreebankTags.indexOf(pos)] += 1;
	}
	
	static void finalizeDistinctPosCount() {
		for(int i = 0; i < pennTreebankTags.size(); i++) {
			int count = 0;
			for(int j = 0; j < pennTreebankTags.size(); j++) {
				if(distinctPosTags[i][j] > 0) {
					count++;
				}
			}
			distinctPosCount[i] = count;
		}
	}
	
	static void updateDistinctWordCount(String pos, String word) {
		
		String newWord = pos + "~~~" + word;
		
		if(distinctWordDictionary.containsKey(newWord)) {
			if(distinctWordCount.containsKey(pos)) {
				int count = distinctWordCount.get(pos);
				count++;
				distinctWordCount.put(pos, count);
			} else {
				distinctWordCount.put(pos, 1);
			}
		} 
		distinctWordDictionary.put(newWord, pos);
	}

	static void updateTotalCount(String word) {

		int currentCount = 0;
		
		if(totalCount.containsKey(word)) {
			currentCount = totalCount.get(word);	
		} 
	
		currentCount++;
		totalCount.put(word, currentCount);
	}
	
	static void updateConditionalCount(String word) {
		
		int currentCount = 0;
		
		if(conditionalCount.containsKey(word)) {
			currentCount = conditionalCount.get(word);	
		} 
	
		currentCount++;
		conditionalCount.put(word, currentCount);
	}
	
	/**
	 * GENERAL METHODS FOR COUNTING POS AND WORDS
	 * -----------------------------------------------------------------------
	 */
	
	/**
	 * ----------------------------------------------------------------------------
	 * METHODS FOR CALCULATING ADD-ONE SMOOTHING
	 */
	static void addOneSmoothing() {
		Enumeration<String> keys = conditionalCount.keys();
		double totalPos = 0;
		double totalWords = 0;
		while(keys.hasMoreElements()) {
			String key = keys.nextElement();
			String[] token = key.split("~~~");

			double count = conditionalCount.get(key); //Perform counting for known words
			double total = totalCount.get(token[0]);
			double probability = count/total;
			transitionalProbabilities.put(key, probability);
			
			if(pennTreebankTags.contains(token[1])) {
				totalPos += total;
			} else {
				totalWords += total;
			}
		}
		
		String posUnknown = "<unknown>~~~pos";
		String wordUnknown = "<unknown>~~~word";
		
		double posUnknownProbability = 1/(totalPos+1); //Add in probability for unknown words
		double wordUnknownProbability = 1/(totalWords+1);
		
		transitionalProbabilities.put(posUnknown, posUnknownProbability);
		transitionalProbabilities.put(wordUnknown, wordUnknownProbability);
	}
	
	static void saveToModelFile() {
		Enumeration<String> pos = transitionalProbabilities.keys();
		try {
			File file = new File(modelFile);
			 
			if (!file.exists()) {
				file.createNewFile();
			}
	
			FileWriter fw = new FileWriter(file.getAbsoluteFile());
			BufferedWriter bw = new BufferedWriter(fw);
			
			while(pos.hasMoreElements()) {
				String key = pos.nextElement();
				bw.write(key + "~~~" + transitionalProbabilities.get(key) + "\n");
			} 
	
			bw.close();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * METHODS FOR CALCULATING ADD-ONE SMOOTHING
	 * ----------------------------------------------------------------------------
	 */
	
	/**
	 * ----------------------------------------------------------------------------
	 * METHODS FOR CALCULATING WITTEN BELL SMOOTHING (NOTE: THIS METHODS ARE NOT USED AS AFTER VALIDATION, THEY PRODUCE WORSE RESULTS)
	 */
	static void performWittenBellSmoothing() {
		//Perform smoothing for POS tags
		for(int i = 0; i < pennTreebankTags.size(); i++) {
			for(int j = 0; j < pennTreebankTags.size(); j++) {
				String pos = pennTreebankTags.get(i) + "~~~" + pennTreebankTags.get(j);
				double probability = 0.0;
				if(conditionalCount.containsKey(pos)) { //C(Wk-1Wk) > 0
					double countOfWordFollowingPos = conditionalCount.get(pos);
					double distinctPos = distinctPosCount[i];
					double total = totalCount.get(pennTreebankTags.get(i));
					probability = countOfWordFollowingPos / (distinctPos + total);
				} else { //C(Wk-1W) = 0
					double distinctPos = distinctPosCount[i];
					double unseen = pennTreebankTags.size() - distinctPos;
					double total = totalCount.get(pennTreebankTags.get(i));
					probability = distinctPos / (unseen*(total + distinctPos));
				}
				transitionalProbabilitiesPos.put(pos, probability);
			}
		}
		
		//Perform smoothing for words
		Enumeration<String> words = distinctWordDictionary.keys();
		//C(Wk-1Wk) > 0
		while(words.hasMoreElements()) {
			double probability = 0.0;
			String key = words.nextElement();
			String[] tokens = key.split("~~~");
			String pos = tokens[0];
			if(conditionalCount.containsKey(key)) {
				double countOfWordFollowingPos = conditionalCount.get(key);
				double distinctWords = distinctWordCount.get(pos);
				double total = totalCount.get(pos);
				probability = countOfWordFollowingPos / (distinctWords + total);
				transitionalProbabilitiesWords.put(key, probability);
			} 
		}
		
		//C(Wk-1W) = 0
		for(int i = 0; i < pennTreebankTagsNoStart.size(); i++) {
			String pos = pennTreebankTags.get(i);
			double probability = 0.0;
			
			double distinctWords = distinctWordCount.get(pos);
			double unseen = totalCount.size() - distinctWords;
	
			double total = totalCount.get(pos);
			probability = distinctWords / (unseen * (total + distinctWords));
			transitionalProbabilitiesWords.put("<unknown>~~~"+pos, probability);
		}
		
		
	}
	
	static void calculateProbabilities(String filename) {
		Enumeration<String> keys = conditionalCount.keys();
		while(keys.hasMoreElements()) {
			String key = keys.nextElement();
			String[] token = key.split("~~~");

			double count = conditionalCount.get(key);
			double total = totalCount.get(token[0]);
			double probability = count/total;
			transitionalProbabilities.put(key, probability);
		}

		Enumeration<String> pos = transitionalProbabilities.keys();
		try {
			File file = new File(filename);
			 
			if (!file.exists()) {
				file.createNewFile();
			}
	
			FileWriter fw = new FileWriter(file.getAbsoluteFile(), true);
			BufferedWriter bw = new BufferedWriter(fw);
			
			while(pos.hasMoreElements()) {
				String key = pos.nextElement();
				bw.write(key + "~~~" + transitionalProbabilities.get(key) + "\n");
			} 
	
			bw.close();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	
	/**
	 * Save probabilities to model_file
	 */
	static void saveProbabilities(String filename) {
		Enumeration<String> pos = transitionalProbabilitiesPos.keys();
		Enumeration<String> words = transitionalProbabilitiesWords.keys();
		try {
			File file = new File(filename);
			 
			if (!file.exists()) {
				file.createNewFile();
			}
	
			FileWriter fw = new FileWriter(file.getAbsoluteFile());
			BufferedWriter bw = new BufferedWriter(fw);
			
			while(pos.hasMoreElements()) {
				String key = pos.nextElement();
				bw.write(key + "~~~" + transitionalProbabilitiesPos.get(key) + "\n");
			} 
			
			while(words.hasMoreElements()) {
				String key = words.nextElement();
				bw.write(key + "~~~" + transitionalProbabilitiesWords.get(key) + "\n");
			} 
			
			bw.close();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * METHODS FOR CALCULATING WITTEN BELL SMOOTHING
	 * ----------------------------------------------------------------------------
	 */
	
	/**
	 * ----------------------------------------------------------------------------
	 * METHODS FOR PERFORMING 10-FOLD CROSS VALIDATION
	 */
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
	
	static ArrayList<ArrayList<String>> tenfoldCrossvalidationSplit(List<String> sentences) {
		int numOfSentence = sentences.size();
		int binSize = numOfSentence / 11;
		int lastBin = numOfSentence % 11 + binSize;
		
		ArrayList<ArrayList<String>> bins = new ArrayList<ArrayList<String>>();
		int currentCount = 0;
		for(int i = 0; i < 10; i++) {
			ArrayList<String> currentBin = new ArrayList<String>();
			
			if(i < 9) {
				for(int j = 0; j < binSize; j++) {
					currentBin.add(sentences.get(currentCount));
					currentCount++;
				}
			} else {
				for(int j = 0; j < lastBin; j++) {
					currentBin.add(sentences.get(currentCount));
					currentCount++;
				}
			}
			bins.add(currentBin);
		}
		
		return bins;
	}
	
	static void performTenfoldCrossvalidation(ArrayList<ArrayList<String>> bins) {

		for(int i = 0; i < 10; i++) {
			conditionalCount = new Hashtable<String, Integer>();
			totalCount = new Hashtable<String, Integer>();
			
			transitionalProbabilitiesPos = new Hashtable<String, Double>();
			transitionalProbabilitiesWords = new Hashtable<String, Double>();
			
			distinctPosTags = new int[pennTreebankTags.size()][pennTreebankTags.size()];
			distinctWordDictionary = new Hashtable<String, String>();
			
			distinctPosCount = new int[pennTreebankTags.size()];
			distinctWordCount = new Hashtable<String, Integer>();
			
			transitionalProbabilities = new Hashtable<String, Double>();

			for(int j = 0; j < 10; j++) {
				if(i == j) { //Current test set
					printTestSetForTenFold(bins.get(j), j);
					printAnswerSetForTenFold(bins.get(j), j);
				}
				else {
					ArrayList<String> currentBin = bins.get(j);
					countPosAndWords(currentBin);

				}
			}
			finalizeDistinctPosCount();
			addOneSmoothing();
			printModelFileForTenFold(i);
		}	
	}

	static void countPosAndWords(List<String> bin) {
		String[] tokens;
		String previousPos = "";
		for(String sentence : bin) {
			sentence = "<s> " + sentence + " </s>";
			tokens = sentence.split(" ");
			
			for(String token : tokens) {
				if(token.equals("<s>")) {
					previousPos = "<s>";
					updateTotalCount(token);
					
				} else if (token.equals("</s>")) {
					updateTotalCount(token);
					updateDistinctPosCount(token, previousPos);
					updateConditionalCount(token + "~~~" + previousPos);
				} else {
					String[] posTokens = token.split("/");
					String word = "";
					String pos = "";
					
					pos = posTokens[posTokens.length-1];
					word = posTokens[0];
					for(int i = 1; i < posTokens.length-1; i++) {
						word += "/" + posTokens[i];
					}
					
					updateTotalCount(pos);
					updateTotalCount(word);
					updateDistinctWordCount(pos, word);
					updateConditionalCount(pos + "~~~" + word); 
					updateDistinctPosCount(pos, previousPos);
					updateConditionalCount(previousPos + "~~~" + pos);
					previousPos = pos;
				}
			}
		}
	}
	
	static void printAnswerSetForTenFold(ArrayList<String> testSet, int index) {
		try {
			File file = new File("TenfoldAnswer" + index);
			 
			if (!file.exists()) {
				file.createNewFile();
			}
	
			FileWriter fw = new FileWriter(file.getAbsoluteFile());
			BufferedWriter bw = new BufferedWriter(fw);
			
			for(String sentence : testSet) {
				bw.write(sentence + "\n");
			} 
	
			bw.close();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	static void printTestSetForTenFold(ArrayList<String> answerSet, int index) {
		try {
			File file = new File("Tenfoldtest" + index);
			 
			if (!file.exists()) {
				file.createNewFile();
			}
	
			FileWriter fw = new FileWriter(file.getAbsoluteFile());
			BufferedWriter bw = new BufferedWriter(fw);
			
			for(String sentence : answerSet) {
				
				String[] posTokens = sentence.split(" ");
				String newSentence = "";
				
				for(String token : posTokens) {
					String[] tokens = token.split("/");

					String word = tokens[0];
					for(int i = 1; i < tokens.length-1; i++) {
						word += "/" + tokens[i];
					}
					newSentence += word + " ";	
				}
				bw.write(newSentence + "\n");
			} 
	
			bw.close();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	static void printModelFileForTenFold(int index) {
		Enumeration<String> pos = transitionalProbabilities.keys();
		try {
			File file = new File("Tenfoldmodel" + index);
			 
			if (!file.exists()) {
				file.createNewFile();
			}
	
			FileWriter fw = new FileWriter(file.getAbsoluteFile());
			BufferedWriter bw = new BufferedWriter(fw);
			
			while(pos.hasMoreElements()) {
				String key = pos.nextElement();
				bw.write(key + "~~~" + transitionalProbabilities.get(key) + "\n");
			} 
	
			bw.close();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/**
	 * METHODS FOR PERFORMING 10-FOLD CROSS VALIDATION
	 * ----------------------------------------------------------------------------
	 */
}

