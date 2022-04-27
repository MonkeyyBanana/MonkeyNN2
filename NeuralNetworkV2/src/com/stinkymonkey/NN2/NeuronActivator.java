package com.stinkymonkey.NN2;

public enum NeuronActivator {
	Identity("Identity", 0, "NONE"),
	Rectifier("Recifier", 1, ""),
	Sin("Sine", 2, "Sine 0.01745240"),
	Cos("Cosine", 3, "Cosine 0.99984769"),
	Tan("Tangent", 4, "Tangent 0.01745506"),
	Tanh("Tangent Hyperbolic", 5, "Tangent Hyperbolic"),
	Sinh("Sine Hyperbolic", 6, "Sine Hyperbolic"),
	Exp("Exponent", 7, "Exponent"),
	Sigmoid("Sigmoid", 8, "Sigmoid"),
	Sqrt("Square Root", 9, "Square Root"),
	Pow2("Squared", 10, "Powered By Two");
		
	private String funcType;
	private int funcId;
	private String funcDesc;
	
	NeuronActivator(String fT, int id, String desc) {
		funcType = fT;
		funcId = id;
		funcDesc = desc;
	}
	
	public String getName() {return funcType;}
	public int getId() {return funcId;}
	public String getDesc() {return funcDesc;}
	
	
	public float activator(float x) {
		float out = 0.0f;
		switch(this.funcId) {
		case 0:
			out = Identity(x);
			break;
		case 1:
			out = Rectifier(x);
			break;
		case 2:
			out = Sin(x);
			break;
		case 3:
			out = Cos(x);
			break;
		case 4:
			out = Tan(x);
			break;
		case 5:
			out = TanH(x);
			break;
		case 6:
			out = SinH(x);
			break;
		case 7:
			out = Exponential(x);
			break;
		case 8:
			out = Sigmoid(x);
			break;
		case 9:
			out = Sqrt(x);
			break;
		case 10:
			out = Pow2(x);
			break;
		}
		return out;
	}
	
	private float Identity(float x) {
		return x;
	}
	
	private float Rectifier(float x) {
		if (x != x) 
			return 0.0f;
		if (x < 0.0f)
			return 0.0f;
		if (x > 1.0f)
			return 1.0f;
		return x;
	}
	
	private float Exponential(float x) {
		float out = (float) Math.exp(x);
		if (x != out)
			return 0.0f;
		return out;
	}
	
	private float Sin(float x) {
		float out = (float) Math.sin(x);
		if (out != out)
			return 0.0f;
		if (out < 0.0f)
			return 0.0f;
		if (out > 1.0f)
			return 1.0f;
		return out;
	}
	
	private float Cos(float x) {
		float out = (float) Math.cos(x);
		if (out != out)
			return 0.0f;
		if (out < 0.0f)
			return 0.0f;
		if (out > 1.0f)
			return 1.0f;
		return out;
	}
	
	private float Tan(float x) {
		float out = (float) Math.tan(x);
		if (out != out)
			return 0.0f;
		if (out < 0.0f)
			return 0.0f;
		if (out > 1.0f)
			return 1.0f;
		return out;
	}
	
	private float TanH(float x) {
		float out = (float) Math.tanh(x);
		if (out != out)
			return 0.0f;
		if (out < 0.0f)
			return 0.0f;
		if (out > 1.0f)
			return 1.0f;
		return out;
	}
	
	private float SinH(float x) {
		float out = (float) Math.sinh(x);
		if (out != out)
			return 0.0f;
		if (out < 0.0f)
			return 0.0f;
		if (out > 1.0f)
			return 1.0f;
		return out;
	}
	
	private float Sigmoid(float x) {
		float out = 1.0f / (1.0f + (float)Math.exp(-x));
		if (out != out)
			return 0.0f;
		if (out < 0.0f)
			return 0.0f;
		if (out > 1.0f)
			return 1.0f;
		return out;
	}
	
	private float Sqrt(float x) {
		float out = (float) Math.sqrt(x);
		if (out != out)
			return 0.0f;
		if (out < 0.0f)
			return 0.0f;
		if (out > 1.0f)
			return 1.0f;
		return out;
	}
	
	private float Pow2(float x) {
		float out = (float) Math.pow(x, 2);
		if (out != out)
			return 0.0f;
		if (out < 0.0f)
			return 0.0f;
		if (out > 1.0f)
			return 1.0f;
		return out;
	}
}
