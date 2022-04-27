package com.stinkymonkey.NN2;

public class NetworkEvolver implements Adjusters {
	Adjusters adjuster;

	public NetworkEvolver(Adjusters adjust) {
		adjuster = adjust;
	}
	
	@Override
	public boolean TrainStreamNextData(float[][] input, float[][] target) {
		return adjuster.TrainStreamNextData(input, target);
	}

	@Override
	public void EvolverProcessOILoss(NetworkProgram src) {
		adjuster.EvolverProcessOILoss(src);
	}

	@Override
	public void EvolverReachedNextGoalEvent() {
		adjuster.EvolverReachedNextGoalEvent();
	}
	
	
	
}
