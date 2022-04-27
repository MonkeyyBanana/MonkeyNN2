package com.stinkymonkey.NN2;

public interface Adjusters {
	public boolean TrainStreamNextData(float[][] input, float[][] target);
	public void EvolverProcessOILoss(NetworkProgram src);
	public void EvolverReachedNextGoalEvent();
}
