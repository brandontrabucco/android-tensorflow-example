package com.btrabucco.tensorflowtest;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;

import org.tensorflow.Tensor;

import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    private float[] compute() {
        TensorFlowEngine tf = new TensorFlowEngine(this, "tf_graph.proto");
        tf.init();
        Tensor input_tensor = Tensor.create(new float[][] {{1.0f}, {0.0f}, {0.0f}});
        Tensor label_tensor = Tensor.create(new float[][] {{1.0f}, {0.0f}, {0.0f}});
        for (int i = 0; i < 10000; i++) {
            tf.train(input_tensor, label_tensor);
        }
        return tf.getOutput();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        final TextView message = (TextView) findViewById(R.id.message);
        message.setText(Arrays.toString(compute()));
    }

}
