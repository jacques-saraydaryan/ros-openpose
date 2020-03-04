// ------------------------- OpenPose Library Tutorial - Real Time Pose Estimation -------------------------
// If the user wants to learn to use the OpenPose library, we highly recommend to start with the `examples/tutorial_*/` folders.
// This example summarizes all the funcitonality of the OpenPose library:
    // 1. Read folder of images / video / webcam  (`producer` module)
    // 2. Extract and render body keypoint / heatmap / PAF of that image (`pose` module)
    // 3. Extract and render face keypoint / heatmap / PAF of that image (`face` module)
    // 4. Save the results on disc (`filestream` module)
    // 5. Display the rendered pose (`gui` module)
    // Everything in a multi-thread scenario (`thread` module)
    // Points 2 to 5 are included in the `wrapper` module
// In addition to the previous OpenPose modules, we also need to use:
    // 1. `core` module:
        // For the Array<float> class that the `pose` module needs
        // For the Datum struct that the `thread` module sends between the queues
    // 2. `utilities` module: for the error & logging functions, i.e. op::error & op::log respectively
// This file should only be used for the user to take specific examples.

// C++ std library dependencies
#include <atomic>
#include <chrono> // `std::chrono::` functions and classes, e.g. std::chrono::milliseconds
#include <cstdio> // sscanf
#include <cstdlib>
#include <string>
//#include <thread> // std::this_thread
#include <vector>
// Other 3rdpary depencencies
#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string

// OpenPose dependencies
// Option a) Importing all modules
#include <openpose/headers.hpp>
// Option b) Manually importing the desired modules. Recommended if you only intend to use a few modules.
// #include <openpose/core/headers.hpp>
// #include <openpose/experimental/headers.hpp>
#include <openpose/face/headers.hpp>
// #include <openpose/filestream/headers.hpp>
// #include <openpose/gui/headers.hpp>
// #include <openpose/pose/headers.hpp>
// #include <openpose/producer/headers.hpp>
// #include <openpose/thread/headers.hpp>
// #include <openpose/utilities/headers.hpp>
// #include <openpose/wrapper/headers.hpp>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <openpose_ros_msgs/Persons.h>
#include <openpose_ros_msgs/BodyPartDetection.h>
#include <openpose_ros_msgs/PersonDetection.h>
#include "openpose_ros_common.hpp"

#include "openpose_ros_srvs/DetectPeoplePoseFromImg.h"

image_transport::Publisher publish_result;
ros::Publisher publish_pose;
ros::ServiceServer pose_srv;

// See all the available parameter options withe the `--help` flag. E.g. `./build/examples/openpose/openpose.bin --help`.
// Note: This command will show you flags for other unnecessary 3rdparty files. Check only the flags for the OpenPose
// executable. E.g. for `openpose.bin`, look for `Flags from examples/openpose/openpose.cpp:`.
// Debugging
DEFINE_int32(logging_level,             3,              "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
                                                        " 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
                                                        " low priority messages and 4 for important ones.");
// Producer
DEFINE_int32(camera,                    0,              "The camera index for cv::VideoCapture. Integer in the range [0, 9].");
DEFINE_double(camera_fps,               30.0,           "Frame rate for the webcam (only used when saving video from webcam). Set this value to the"
                                                        " minimum value between the OpenPose displayed speed and the webcam real frame rate.");
DEFINE_string(video,                    "",             "Use a video file instead of the camera. Use `examples/media/video.avi` for our default"
                                                        " example video.");
DEFINE_string(image_dir,                "",             "Process a directory of images. Use `examples/media/` for our default example folder with 20"
                                                        " images.");
DEFINE_bool(frame_flip,                 false,          "Flip/mirror each frame (e.g. for real time webcam demonstrations).");
DEFINE_int32(frame_rotate,              0,              "Rotate each frame, 4 possible values: 0, 90, 180, 270.");
DEFINE_bool(frames_repeat,              false,          "Repeat frames when finished.");
// OpenPose
DEFINE_string(model_folder,             std::string(std::getenv("OPENPOSE_HOME")) + std::string("/models/"),      "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(resolution,               "640x480",     "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
                                                        " default images resolution.");
DEFINE_int32(num_gpu,                   -1,             "The number of GPU devices to use. If negative, it will use all the available GPUs in your"
                                                        " machine.");
DEFINE_int32(num_gpu_start,             0,              "GPU device start number.");
DEFINE_int32(keypoint_scale,            0,              "Scaling of the (x,y) coordinates of the final pose data array, i.e. the scale of the (x,y)"
                                                        " coordinates that will be saved with the `write_keypoint` & `write_keypoint_json` flags."
                                                        " Select `0` to scale it to the original source resolution, `1`to scale it to the net output"
                                                        " size (set with `net_resolution`), `2` to scale it to the final output size (set with"
                                                        " `resolution`), `3` to scale it in the range [0,1], and 4 for range [-1,1]. Non related"
                                                        " with `num_scales` and `scale_gap`.");
// OpenPose Body Pose
DEFINE_string(model_pose,               "BODY_25",         "Model to be used (e.g. COCO, MPI, MPI_4_layers).");
//DEFINE_string(net_resolution,           "320x320",      "Multiples of 16. If it is increased, the accuracy usually increases. If it is decreased,"
 //                                                       " the speed increases.");

DEFINE_string(net_resolution,           "-1x368",       "Multiples of 16. If it is increased, the accuracy potentially increases. If it is"
                                                        " decreased, the speed increases. For maximum speed-accuracy balance, it should keep the"
                                                        " closest aspect ratio possible to the images or videos to be processed. Using `-1` in"
                                                        " any of the dimensions, OP will choose the optimal aspect ratio depending on the user's"
                                                        " input value. E.g. the default `-1x368` is equivalent to `656x368` in 16:9 resolutions,"
                                                        " e.g. full HD (1980x1080) and HD (1280x720) resolutions.");



DEFINE_int32(num_scales,                1,              "Number of scales to average.");
DEFINE_double(scale_gap,                0.3,            "Scale gap between scales. No effect unless num_scales>1. Initial scale is always 1. If you"
                                                        " want to change the initial scale, you actually want to multiply the `net_resolution` by"
                                                        " your desired initial scale.");
DEFINE_bool(heatmaps_add_parts,         false,          "If true, it will add the body part heatmaps to the final op::Datum::poseHeatMaps array"
                                                        " (program speed will decrease). Not required for our library, enable it only if you intend"
                                                        " to process this information later. If more than one `add_heatmaps_X` flag is enabled, it"
                                                        " will place then in sequential memory order: body parts + bkg + PAFs. It will follow the"
                                                        " order on POSE_BODY_PART_MAPPING in `include/openpose/pose/poseParameters.hpp`.");
DEFINE_bool(heatmaps_add_bkg,           false,          "Same functionality as `add_heatmaps_parts`, but adding the heatmap corresponding to"
                                                        " background.");
DEFINE_bool(heatmaps_add_PAFs,          false,          "Same functionality as `add_heatmaps_parts`, but adding the PAFs.");
// OpenPose Face
DEFINE_bool(face,                       false,          "Enables face keypoint detection. It will share some parameters from the body pose, e.g."
                                                        " `model_folder`.");
DEFINE_string(face_net_resolution,      "160x160",      "Multiples of 16. Analogous to `net_resolution` but applied to the face keypoint detector."
                                                        " 320x320 usually works fine while giving a substantial speed up when multiple faces on the"
                                                        " image.");

// OpenPose Rendering Face
DEFINE_double(face_render_threshold,    0.4,            "Analogous to `render_threshold`, but applied to the face keypoints.");
DEFINE_int32(face_render,               -1,             "Analogous to `render_pose` but applied to the face. Extra option: -1 to use the same"
                                                        " configuration that `render_pose` is using.");
DEFINE_double(face_alpha_pose,          0.6,            "Analogous to `alpha_pose` but applied to face.");
DEFINE_double(face_alpha_heatmap,       0.7,            "Analogous to `alpha_heatmap` but applied to face.");







// OpenPose Hand
DEFINE_bool(hand,                       false,          "Enables hand keypoint detection. It will share some parameters from the body pose, e.g."
                                                        " `model_folder`.");
DEFINE_string(hand_net_resolution,      "480x480",      "Multiples of 16. Analogous to `net_resolution` but applied to the hand keypoint detector.");
// OpenPose Rendering
DEFINE_int32(part_to_show,              0,              "Part to show from the start.");
DEFINE_bool(disable_blending,           false,          "If blending is enabled, it will merge the results with the original frame. If disabled, it"
                                                        " will only display the results.");
// OpenPose Rendering Pose
DEFINE_int32(render_pose,               1,              "Set to 0 for no rendering, 1 for CPU rendering (slightly faster), and 2 for GPU rendering"
                                                        " (slower but greater functionality, e.g. `alpha_X` flags). If rendering is enabled, it will"
                                                        " render both `outputData` and `cvOutputData` with the original image and desired body part"
                                                        " to be shown (i.e. keypoints, heat maps or PAFs).");
DEFINE_double(alpha_pose,               0.6,            "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
                                                        " hide it. Only valid for GPU rendering.");
DEFINE_double(alpha_heatmap,            0.7,            "Blending factor (range 0-1) between heatmap and original frame. 1 will only show the"
                                                        " heatmap, 0 will only show the frame. Only valid for GPU rendering.");
// OpenPose Rendering Face
DEFINE_int32(render_face,               -1,             "Analogous to `render_pose` but applied to the face. Extra option: -1 to use the same"
                                                        " configuration that `render_pose` is using.");
DEFINE_double(alpha_face,               0.6,            "Analogous to `alpha_pose` but applied to face.");
DEFINE_double(alpha_heatmap_face,       0.7,            "Analogous to `alpha_heatmap` but applied to face.");
// OpenPose Rendering Hand
DEFINE_int32(render_hand,               -1,             "Analogous to `render_pose` but applied to the hand. Extra option: -1 to use the same"
                                                        " configuration that `render_pose` is using.");
DEFINE_double(alpha_hand,               0.6,            "Analogous to `alpha_pose` but applied to hand.");
DEFINE_double(alpha_heatmap_hand,       0.7,            "Analogous to `alpha_heatmap` but applied to hand.");
// Display
DEFINE_bool(fullscreen,                 false,          "Run in full-screen mode (press f during runtime to toggle).");
DEFINE_bool(process_real_time,          false,          "Enable to keep the original source frame rate (e.g. for video). If the processing time is"
                                                        " too long, it will skip frames. If it is too fast, it will slow it down.");
DEFINE_bool(no_gui_verbose,             false,          "Do not write text on output images on GUI (e.g. number of current frame and people). It"
                                                        " does not affect the pose rendering.");
DEFINE_bool(no_display,                 false,          "Do not open a display window.");
// Result Saving
DEFINE_string(write_images,             "",             "Directory to write rendered frames in `write_images_format` image format.");
DEFINE_string(write_images_format,      "png",          "File extension and format for `write_images`, e.g. png, jpg or bmp. Check the OpenCV"
                                                        " function cv::imwrite for all compatible extensions.");
DEFINE_string(write_video,              "",             "Full file path to write rendered frames in motion JPEG video format. It might fail if the"
                                                        " final path does not finish in `.avi`. It internally uses cv::VideoWriter.");
DEFINE_string(write_keypoint,           "",             "Directory to write the people body pose keypoint data. Set format with `write_keypoint_format`.");
DEFINE_string(write_keypoint_format,    "yml",          "File extension and format for `write_keypoint`: json, xml, yaml & yml. Json not available"
                                                        " for OpenCV < 3.0, use `write_keypoint_json` instead.");
DEFINE_string(write_keypoint_json,      "",             "Directory to write people pose data in *.json format, compatible with any OpenCV version.");
DEFINE_string(write_coco_json,          "",             "Full file path to write people pose data with *.json COCO validation format.");
DEFINE_string(write_heatmaps,           "",             "Directory to write heatmaps in *.png format. At least 1 `add_heatmaps_X` flag must be"
                                                        " enabled.");
DEFINE_string(write_heatmaps_format,    "png",          "File extension and format for `write_heatmaps`, analogous to `write_images_format`."
                                                        " Recommended `png` or any compressed and lossless format.");

DEFINE_string(result_image_topic,              "",          "topic name for publish processed/annotated image(usefule for debugging)");


DEFINE_double(render_threshold,         0.05,           "Only estimated keypoints whose score confidences are higher than this threshold will be"
                                                        " rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
                                                        " while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
                                                        " more false positives (i.e. wrong detections).");
DEFINE_int32(scale_number,              1,              "Number of scales to average.");

DEFINE_string(output_resolution,        "-1x-1",        "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
                                                        " input image resolution.");


op::PoseModel gflagToPoseModel(const std::string& poseModeString)
{
    // It will print info if desiredPriority >= sPriorityThreshold
    // opLog(message, desiredPriority, __LINE__, __FUNCTION__, __FILE__);
    op::opLog("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    if (poseModeString == "COCO")
        return op::PoseModel::COCO_18;
    else if (poseModeString == "MPI")
        return op::PoseModel::MPI_15;
    else if (poseModeString == "MPI_4_layers")
        return op::PoseModel::MPI_15_4;
    else
    {
        op::error("String does not correspond to any model (COCO, MPI, MPI_4_layers)", __LINE__, __FUNCTION__, __FILE__);
        return op::PoseModel::COCO_18;
    }
}

op::ScaleMode gflagToScaleMode(const int keypointScale)
{
    op::opLog("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    if (keypointScale == 0)
        return op::ScaleMode::InputResolution;
    else if (keypointScale == 1)
        return op::ScaleMode::NetOutputResolution;
    else if (keypointScale == 2)
        return op::ScaleMode::OutputResolution;
    else if (keypointScale == 3)
        return op::ScaleMode::ZeroToOne;
    else if (keypointScale == 4)
        return op::ScaleMode::PlusMinusOne;
    else
    {
        const std::string message = "String does not correspond to any scale mode: (0, 1, 2, 3, 4) for (InputResolution,"
                                    " NetOutputResolution, OutputResolution, ZeroToOne, PlusMinusOne).";
        op::error(message, __LINE__, __FUNCTION__, __FILE__);
        return op::ScaleMode::InputResolution;
    }
}

std::vector<op::HeatMapType> gflagToHeatMaps(const bool heatMapsAddParts, const bool heatMapsAddBkg, const bool heatMapsAddPAFs)
{
    std::vector<op::HeatMapType> heatMapTypes;
    if (heatMapsAddParts)
        heatMapTypes.emplace_back(op::HeatMapType::Parts);
    if (heatMapsAddBkg)
        heatMapTypes.emplace_back(op::HeatMapType::Background);
    if (heatMapsAddPAFs)
        heatMapTypes.emplace_back(op::HeatMapType::PAFs);
    return heatMapTypes;
}

op::RenderMode gflagToRenderMode(const int renderFlag, const int renderPoseFlag = -2)
{
    if (renderFlag == -1 && renderPoseFlag != -2)
        return gflagToRenderMode(renderPoseFlag, -2);
    else if (renderFlag == 0)
        return op::RenderMode::None;
    else if (renderFlag == 1)
        return op::RenderMode::Cpu;
    else if (renderFlag == 2)
        return op::RenderMode::Gpu;
    else
    {
        op::error("Undefined RenderMode selected.", __LINE__, __FUNCTION__, __FILE__);
        return op::RenderMode::None;
    }
}

// Google flags into program variables
std::tuple<op::Point<int>, op::Point<int>, op::Point<int>, op::Point<int>, op::PoseModel, op::ScaleMode,
           std::vector<op::HeatMapType>> gflagsToOpParameters()
{
    op::opLog("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    // outputSize
    op::Point<int> outputSize;
    auto nRead = sscanf(FLAGS_resolution.c_str(), "%dx%d", &outputSize.x, &outputSize.y);
    nRead = sscanf(FLAGS_resolution.c_str(), "%dx%d", &outputSize.x, &outputSize.y);
    op::checkEqual(nRead, 2, "Error, resolution format (" +  FLAGS_resolution + ") invalid, should be e.g., 960x540 ",
               __LINE__, __FUNCTION__, __FILE__);
    // netInputSize
    op::Point<int> netInputSize;
    nRead = sscanf(FLAGS_net_resolution.c_str(), "%dx%d", &netInputSize.x, &netInputSize.y);
    op::checkEqual(nRead, 2, "Error, net resolution format (" +  FLAGS_net_resolution + ") invalid, should be e.g., 656x368 (multiples of 16)",
               __LINE__, __FUNCTION__, __FILE__);
    // faceNetInputSize
    op::Point<int> faceNetInputSize;
    nRead = sscanf(FLAGS_face_net_resolution.c_str(), "%dx%d", &faceNetInputSize.x, &faceNetInputSize.y);
    op::checkEqual(nRead, 2, "Error, face net resolution format (" +  FLAGS_face_net_resolution
               + ") invalid, should be e.g., 368x368 (multiples of 16)", __LINE__, __FUNCTION__, __FILE__);
    // handNetInputSize
    op::Point<int> handNetInputSize;
    nRead = sscanf(FLAGS_hand_net_resolution.c_str(), "%dx%d", &handNetInputSize.x, &handNetInputSize.y);
    op::checkEqual(nRead, 2, "Error, hand net resolution format (" +  FLAGS_hand_net_resolution
               + ") invalid, should be e.g., 368x368 (multiples of 16)", __LINE__, __FUNCTION__, __FILE__);
    // poseModel
    const auto poseModel = gflagToPoseModel(FLAGS_model_pose);
    // keypointScale
    const auto keypointScale = gflagToScaleMode(FLAGS_keypoint_scale);
    // heatmaps to add
    const auto heatMapTypes = gflagToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg, FLAGS_heatmaps_add_PAFs);
    // Return
    return std::make_tuple(outputSize, netInputSize, faceNetInputSize, handNetInputSize, poseModel, keypointScale, heatMapTypes);
}

op::Point<int> outputSize= op::flagsToPoint(op::String(FLAGS_output_resolution), "-1x-1");
op::Point<int> netInputSize= op::flagsToPoint(op::String(FLAGS_net_resolution), "-1x368");
op::Point<int> netOutputSize;
op::Point<int> faceNetInputSize;
op::Point<int> handNetInputSize;
op::ScaleMode keypointScale;
std::vector<op::HeatMapType> heatMapTypes;

const auto poseModel = op::flagsToPoseModel(op::String(FLAGS_model_pose));
op::CvMatToOpInput cvMatToOpInput{poseModel};
op::CvMatToOpOutput *cvMatToOpOutput;
op::PoseExtractorCaffe poseExtractorCaffe{poseModel, FLAGS_model_folder, FLAGS_num_gpu_start};
//op::PoseRenderer *poseRenderer;
op::PoseCpuRenderer poseRenderer{poseModel, (float)FLAGS_render_threshold, !FLAGS_disable_blending,
                                     (float)FLAGS_alpha_pose};
op::ScaleAndSizeExtractor scaleAndSizeExtractor(netInputSize, outputSize, FLAGS_scale_number, FLAGS_scale_gap);

op::FaceDetector *faceDetector;
//op::FaceExtractorNet *faceExtractor;
op::FaceRenderer *faceRenderer;

op::OpOutputToCvMat *opOutputToCvMat;

int init_openpose()
{
    // logging_level
    op::checkBool(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.", __LINE__, __FUNCTION__, __FILE__);
    op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
   

    const auto timerBegin = std::chrono::high_resolution_clock::now();

    // Applying user defined configuration
   
    netOutputSize = netInputSize;

    // Initialize
    
    cvMatToOpOutput = new op::CvMatToOpOutput();
    
    //FIME face not currently tested
   
    //faceDetector = new op::FaceDetector(poseModel);
    //faceExtractor = new op::FaceExtractorCaffe(faceNetInputSize, faceNetInputSize, FLAGS_model_folder, FLAGS_num_gpu_start);
    //faceRenderer = new op::FaceCpuRenderer(FLAGS_face_render_threshold, FLAGS_face_alpha_pose, FLAGS_face_alpha_heatmap);
    
    opOutputToCvMat = new op::OpOutputToCvMat();

    poseExtractorCaffe.initializationOnThread();
    poseRenderer.initializationOnThread();
    //faceExtractor->initializationOnThread();
    //faceRenderer->initializationOnThread();
    

    // Measuring total time
    const auto now = std::chrono::high_resolution_clock::now();
    const auto totalTimeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(now-timerBegin).count() * 1e-9;
    const auto message = "Initialized. Total time: " + std::to_string(totalTimeSec) + " seconds.";
    ROS_INFO_STREAM(message);

    return 0;
}


openpose_ros_msgs::Persons processImgForPoseDetection(cv_bridge::CvImagePtr &cv_ptr){
    ros::Time t = ros::Time::now();
    std::vector<float> scaleRatios;
    // process
    const op::Point<int> imageSize{cv_ptr->image.cols, cv_ptr->image.rows};
    std::vector<double> scaleInputToNetInputs;
    std::vector<op::Point<int>> netInputSizes;
    double scaleInputToOutput;
    op::Point<int> outputResolution;
    std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
        = scaleAndSizeExtractor.extract(imageSize);
    // Step 3 - Format input image to OpenPose input and output formats
  
    const auto netInputArray = cvMatToOpInput.createArray(OP_CV2OPMAT(cv_ptr->image), scaleInputToNetInputs, netInputSizes);

    auto outputArray = cvMatToOpOutput->createArray(OP_CV2OPMAT(cv_ptr->image), scaleInputToOutput, outputResolution);




    // Step 3 - Estimate poseKeypoints

    poseExtractorCaffe.forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
    const auto poseKeypoints = poseExtractorCaffe.getPoseKeypoints();

    //const auto faces = faceDetector->detectFaces(poseKeypoints, scaleInputToOutput);
    //faceExtractor->forwardPass(faces, cv_ptr->image, scaleInputToOutput);
    //const auto faceKeypoints = faceExtractor->getFaceKeypoints();

    


    // publish annotations.
    openpose_ros_msgs::Persons persons;
    persons.rostime = t;
    persons.image_w = outputSize.x;
    persons.image_h = outputSize.y;
    
    const int num_people = poseKeypoints.getSize(0);
    const int num_bodyparts = poseKeypoints.getSize(1);

    for (size_t person_idx = 0; person_idx < num_people; person_idx++) {
        openpose_ros_msgs::PersonDetection person;
        for (size_t bodypart_idx = 0; bodypart_idx < num_bodyparts; bodypart_idx++) {
            size_t final_idx = 3*(person_idx*num_bodyparts + bodypart_idx);
            openpose_ros_msgs::BodyPartDetection bodypart;
            bodypart.part_id = bodypart_idx;
            bodypart.x = poseKeypoints[final_idx];
            bodypart.y = poseKeypoints[final_idx+1];
            bodypart.confidence = poseKeypoints[final_idx+2];
            person.body_part.push_back(bodypart);
        }
        persons.persons.push_back(person);
    }


    // publish result image with annotation.
    if (!FLAGS_result_image_topic.empty()) {

        poseRenderer.renderPose(outputArray, poseKeypoints, scaleInputToOutput);
    // Step 6 - OpenPose output format to cv::Mat
    auto outputImage = opOutputToCvMat->formatToCvMat(outputArray);
        //poseRenderer.renderPose(outputArray, poseKeypoints);
       // faceRenderer->renderFace(outputArray, faceKeypoints);

        //auto outputImage = opOutputToCvMat->formatToCvMat(outputArray);

        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", OP_OP2CVMAT(outputImage)).toImageMsg();
        publish_result.publish(msg);
    }
    return persons;
}


void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    ros::Time t = ros::Time::now();

    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    } catch (cv_bridge::Exception& e) {
        return;
    }
    if (cv_ptr->image.empty()) return;

    processImgForPoseDetection(cv_ptr);

}

bool peoplePoseFromImg(openpose_ros_srvs::DetectPeoplePoseFromImg::Request  &req,
         openpose_ros_srvs::DetectPeoplePoseFromImg::Response &res)
{
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(req.img, "bgr8");
    } catch (cv_bridge::Exception& e) {
        return false;
    }
    if (cv_ptr->image.empty()) return false;
    
    openpose_ros_msgs::Persons persons=processImgForPoseDetection(cv_ptr);
    res.personList=persons;
    return true;
}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "openpose_ros_node");
    ros::NodeHandle local_nh("~");

    FLAGS_resolution = getParam(local_nh, "resolution", std::string("640x480"));
    FLAGS_num_gpu = getParam(local_nh, "num_gpu", -1);
    FLAGS_num_gpu_start = getParam(local_nh, "num_gpu_start", 1);
    FLAGS_model_pose = getParam(local_nh, "model_pose", std::string("BODY_25"));
    //FLAGS_net_resolution = getParam(local_nh, "net_resolution", std::string("640x480"));
    FLAGS_net_resolution = getParam(local_nh, "net_resolution", std::string("-1x480"));
    FLAGS_face = getParam(local_nh, "face", false);
    FLAGS_no_display = getParam(local_nh, "no_display", false);

    std::string camera_src = getParam(local_nh, "camera", std::string("/camera/image"));
    FLAGS_result_image_topic = getParam(local_nh, "result_image_topic", std::string(""));

    // prepare model
    init_openpose();
  
    // subscribe image
    ros::NodeHandle nh;
    image_transport::ImageTransport img_t(nh);
    image_transport::Subscriber sub = img_t.subscribe(camera_src, 1, imageCallback);
    if (!FLAGS_result_image_topic.empty()) {
        publish_result = img_t.advertise(FLAGS_result_image_topic, 1);
    }
    publish_pose = nh.advertise<openpose_ros_msgs::Persons>("/openpose/pose", 1);
    pose_srv = nh.advertiseService("people_pose_from_img", peoplePoseFromImg);

    ros::spin();
    return 0; 
}
