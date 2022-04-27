package com.stinkymonkey.NN2;

public class NetworkProgram {
	public NeuralNetwork srcNetwork;
	public NetworkContext context = new NetworkContext();
	public boolean hasIn;
	public boolean hasOut;
	public int state = -1, total = -1;
	public float loss = 0.0f;
	
	public NetworkProgram(NeuralNetwork src) {
		srcNetwork = src;
		context.Sync(src);
		hasIn = false;
		hasOut = false;
	}
	
	public void RunTask() {
		srcNetwork.RunTask(context);
		hasIn = false;
		hasOut = true;
	}
}
