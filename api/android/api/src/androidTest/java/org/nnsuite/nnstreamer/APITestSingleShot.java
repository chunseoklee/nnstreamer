package org.nnsuite.nnstreamer;

import android.os.Environment;
import android.support.test.rule.GrantPermissionRule;
import android.support.test.runner.AndroidJUnit4;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.File;
import java.nio.ByteBuffer;

import static org.junit.Assert.*;
import android.util.Log;



/**
 * Testcases for SingleShot.
 */
@RunWith(AndroidJUnit4.class)
public class APITestSingleShot {
    @Rule
    public GrantPermissionRule mPermissionRule = APITestCommon.grantPermissions();

    @Before
    public void setUp() {
        APITestCommon.initNNStreamer();
    }
    @Test
    public void testNNFWNLUFixed() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.NNFW)) {
            /* cannot run the test */
            return;
        }

        String root = Environment.getExternalStorageDirectory().getAbsolutePath();
        Log.d("model root", root);
        File model = new File(root + "/nnstreamer/ONE_nlu_model/fixed/sra_enlu_fixed64.q3.opt.circle");

        try {
            SingleShot single = new SingleShot(model, NNStreamer.NNFWType.NNFW);
            TensorsInfo inInfo = single.getInputInfo();
            TensorsInfo outInfo = single.getOutputInfo();

            printTensorsInfo(inInfo, true);
            printTensorsInfo(outInfo, false);

            Log.d("nnstreamer-test", "Try to set input dim");
            single.setInputInfo(inInfo);

            TensorsInfo inResult = single.getInputInfo();
            TensorsInfo outResult = single.getInputInfo();

            printTensorsInfo(inResult, true);
            printTensorsInfo(outResult, false);

            single.close();
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testNNFWNLUDynamic() {
        /* Be carefull, this test requires backend-bcq and should be run without tensorflow-lite. */
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.NNFW)) {
            /* cannot run the test */
            return;
        }

        String root = Environment.getExternalStorageDirectory().getAbsolutePath();
        File model = new File(root + "/nnstreamer/ONE_nlu_model/dynamic/sra_enlu_notfixed.q3.circle");

        try {
            SingleShot single = new SingleShot(model, NNStreamer.NNFWType.NNFW);
            TensorsInfo inInfo = single.getInputInfo();
            TensorsInfo outInfo = single.getOutputInfo();
            final int tokenLen = 64;

            printTensorsInfo(inInfo, true);
            printTensorsInfo(outInfo, false);

            Log.d("nnstreamer-test", "Try to set input dim");
            inInfo.setTensorDimension(0, new int[]{tokenLen,1});
            inInfo.setTensorDimension(1, new int[]{1});
            inInfo.setTensorDimension(2, new int[]{tokenLen,1});
            inInfo.setTensorDimension(3, new int[]{tokenLen,1});
            inInfo.setTensorDimension(4, new int[]{882,tokenLen,1});

            single.setInputInfo(inInfo);

            TensorsInfo inResult = single.getInputInfo();
            TensorsInfo outResult = single.getOutputInfo();

            printTensorsInfo(inResult, true);
            printTensorsInfo(outResult, false);

            single.close();
        } catch (Exception e) {
            fail();
        }
    }

    private void printTensorsInfo(TensorsInfo info, boolean isInput) {
        int num = info.getTensorsCount();

        Log.d("nnstreamertest", "Print " + (isInput ? "input" : "output") + " info");
        Log.d("nnstreamertest", "The number of tensors in info: " + num);
        for (int i = 0; i < num; i++) {
            int[] dim = info.getTensorDimension(i);

            Log.d("nnstreamertest", "Info index " + i +
                    " name: " + info.getTensorName(i) +
                    " type: " + info.getTensorType(i) +
                    " dim: " + dim[0] + ":" + dim[1] + ":" + dim[2] + ":" + dim[3]);
        }
    }


}
