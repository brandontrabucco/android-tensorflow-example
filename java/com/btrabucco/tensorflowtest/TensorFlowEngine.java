package com.btrabucco.tensorflowtest;

import android.app.Activity;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.List;

/**
 * Created by brand on 1/9/2018.
 */

public class TensorFlowEngine {

    private Graph g;
    private Session s;
    private Tensor output_tensor;
    private Tensor loss_tensor;

    static {
        System.loadLibrary("tensorflow_inference");
    }

    public TensorFlowEngine(Activity a, String file) {
        try {
            g = new Graph();
            InputStream initialStream = a.getAssets().open(file);
            byte[] targetArray = new byte[initialStream.available()];
            initialStream.read(targetArray);
            g.importGraphDef(targetArray);
            s = new Session(g);
        } catch (IOException e) {
            e.printStackTrace();
            g = null;
            s = null;
        }
    }

    public void init() {
        s.runner().addTarget("init").run();
    }

    public void inference(Tensor input_tensor) {
        List<Tensor<?>> results = s.runner()
            .feed("input_tensor", input_tensor)
            .fetch("output_tensor")
            .run();
        output_tensor = results.get(0);
    }

    public void train(Tensor input_tensor, Tensor label_tensor) {
        List<Tensor<?>> results = s.runner()
                .feed("input_tensor", input_tensor)
                .feed("label_tensor", label_tensor)
                .addTarget("backprop")
                .fetch("output_tensor")
                .fetch("loss_tensor")
                .run();
        output_tensor = results.get(0);
        loss_tensor = results.get(1);
    }

    public float[] getOutput() {
        FloatBuffer buffer = FloatBuffer.allocate(3);
        output_tensor.writeTo(buffer);
        return buffer.array();
    }

    public float getLoss() {
        return loss_tensor.floatValue();
    }

}
