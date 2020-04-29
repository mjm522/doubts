import numpy as np
from pydrake.geometry import ConnectDrakeVisualizer
from pydrake.geometry.render import (
    DepthCameraProperties,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
)
from pydrake.systems.lcm import LcmPublisherSystem
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.sensors import RgbdSensor, ImageToLcmImageArrayT, PixelType
from robotlocomotion import image_array_t
from pydrake.systems.meshcat_visualizer import (
    MeshcatPointCloudVisualizer,
    MeshcatVisualizer
)
from pydrake.perception import DepthImageToPointCloud

enable_collision=True
enable_camera=True

if enable_collision:
    model_path = './models_old/{0}/sdf/{0}.sdf'
else:
    model_path = './models_old/{0}/sdf/{0}_nc.sdf'

dt_file = model_path.format('plane')
cracker_box_file = model_path.format('003_cracker_box')
tomato_soup_can_file = model_path.format('005_tomato_soup_can')
mustard_bottle_file = model_path.format('006_mustard_bottle')

order = [[3,5,6,6,5], [5,6,5,5,5,5,3], [6,6,5,6]]

poses = [[  {'pos':[-0.15,0.2,0.12], 'ori':[0.,0.,0.]},
            {'pos':[-0.06,0.15,0.06], 'ori':[0.,0.,0.]},
            {'pos':[0.02,0.16,0.11], 'ori':[0.,0.,0.]},
            {'pos':[0.09,0.16,0.11], 'ori':[0.,0.,0.]},
            {'pos':[-0.06,0.23,0.06], 'ori':[0.,0.,0.]}  ],

         [  {'pos':[-0.15,-0.06,0.06], 'ori':[0.,0.,0.]},
            {'pos':[-0.16,0.04,0.11], 'ori':[0.,0.,0.]},
            {'pos':[-0.08,0.02,0.06], 'ori':[0.,0.,0.]},
            {'pos':[-0.07,-0.06,0.06], 'ori':[0.,0.,0.]},
            {'pos':[0.,0.02,0.06], 'ori':[0.,0.,0.]},
            {'pos':[ 0.01,-0.06,0.06], 'ori':[0.,0.,0.]},
            {'pos':[ 0.1,-0.01,0.12], 'ori':[0.,0.,0.]}   ],

         [  {'pos':[-0.16,-0.24,0.11], 'ori':[0.,0.,0.]},
            {'pos':[-0.08,-0.24,0.11], 'ori':[0.,0.,0.]},
            {'pos':[-0.01,-0.25,0.06], 'ori':[0.,0.,0.]},
            {'pos':[ 0.06,-0.24,0.11], 'ori':[0.,0.,0.]} ]
         ]

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
parser = Parser(plant)

def xyz_rpy_deg(xyz, rpy_deg):
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi / 180), xyz)

def get_instance(bag_id, item_id, sku_id):
    idd = "{0}{1}{2}".format(bag_id, item_id, sku_id)
    if sku_id == 6:
        return parser.AddModelFromFile(mustard_bottle_file, 'mustard_bottle'+idd)
    if sku_id == 3:
        return parser.AddModelFromFile(cracker_box_file, 'cracker_box'+idd)
    if sku_id == 5:
        return parser.AddModelFromFile(tomato_soup_can_file, 'tomato_soup'+idd)

def set_object_pose(obj_instance, context, pose):
    obj_idx = plant.GetBodyIndices(obj_instance)
    plant.SetFreeBodyPose(context, plant.get_body(obj_idx.pop()), pose)

dt = parser.AddModelFromFile(dt_file, model_name="plane")
plant.WeldFrames(
    plant.world_frame(), plant.GetFrameByName("plane_base", dt),
    X_AB=xyz_rpy_deg(np.array([0.0, 0.0, 0]), [0, 0, 0]))

if enable_camera:
    class Camera():
        def __init__(self, name, pose):
            self.name = name
            self.pose = xyz_rpy_deg(pose[0:3], pose[3:])
            self.reference_object_frame = "plane_base"
            self.reference_object = "plane"

    camera = Camera('camera', [-0.05, 0., 1.1, 190.0, 0.0, -90.0])
    camera_images_rgb = {}
    camera_images_depth = {}

    depth_prop = DepthCameraProperties(width=640, height=480, fov_y=np.pi/4, renderer_name="renderer", z_near=0.01, z_far=10.)
    world_id = plant.GetBodyFrameIdOrThrow(plant.world_body().index())
    frame_id = None
    if camera.reference_object is not None:
        # get the reference object's body indexes
        body_indices = plant.GetBodyIndices(plant.GetModelInstanceByName(camera.reference_object))
        for idx in body_indices:
            body = plant.get_body(idx)
            if body.name() == camera.reference_object_frame:
                # we found the frame we want
                frame_id = plant.GetBodyFrameIdOrThrow(idx)
    if frame_id is None:
        frame_id = world_id
    camera_instance = RgbdSensor(frame_id, X_PB=camera.pose, color_properties=depth_prop, depth_properties=depth_prop)
    camera_instance.set_name(camera.name)
    builder.AddSystem(camera_instance)
    builder.Connect(scene_graph.get_query_output_port(), camera_instance.query_object_input_port())
    #add to the dictionary of cameras, with no image associated for now
    camera_images_rgb[camera.name] = np.zeros([480,640,4])
    camera_images_depth[camera.name] = np.zeros([480,640,1])
    color_info = camera_instance.color_camera_info()
    depth_info = camera_instance.depth_camera_info()
    cloud_generator_instance = DepthImageToPointCloud(depth_info)
    builder.AddSystem(cloud_generator_instance)
    builder.Connect(camera_instance.depth_image_32F_output_port(), cloud_generator_instance.depth_image_input_port())

    image_to_lcm_image_array = builder.AddSystem(ImageToLcmImageArrayT())
    image_to_lcm_image_array.set_name("converter")
    cam_port = (image_to_lcm_image_array.DeclareImageInputPort[PixelType.kRgba8U]("camera_" + str(0)))
    builder.Connect(camera_instance.color_image_output_port(), cam_port)

    image_array_lcm_publisher = builder.AddSystem(
            LcmPublisherSystem.Make(
            channel="DRAKE_RGBD_CAMERA_IMAGES",
            lcm_type=image_array_t,
            lcm=None,
            publish_period=0.03,
            use_cpp_serializer=True))
    image_array_lcm_publisher.set_name("rgbd_publisher")
    builder.Connect(image_to_lcm_image_array.image_array_t_msg_output_port(), image_array_lcm_publisher.get_input_port(0))

    # viz = builder.AddSystem(MeshcatVisualizer(scene_graph))
    # builder.Connect(scene_graph.get_pose_bundle_output_port(), viz.get_input_port(0))
    # pc_viz = builder.AddSystem(MeshcatPointCloudVisualizer(viz, viz.draw_period))
    # builder.Connect(cloud_generator_instance['camera_hand'].point_cloud_output_port(),  pc_viz.GetInputPort("point_cloud_P"))

sku_intances = [[],[],[]]
for j, bag in enumerate(order):
    for i, sku in enumerate(bag):
        sku_intances[j].append(get_instance(j,i,sku))

plant.Finalize()
scene_graph.AddRenderer("renderer", MakeRenderEngineVtk(RenderEngineVtkParams()))
ConnectDrakeVisualizer(builder, scene_graph)
diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
simulator = Simulator(diagram, diagram_context)
plant_context = diagram.GetMutableSubsystemContext(plant, simulator.get_mutable_context())

for sku_instant, pose in zip(sku_intances, poses):
    for ins, p in zip(sku_instant, pose):
        set_object_pose(ins, plant_context, xyz_rpy_deg(p['pos'], p['ori']))

                     
def camera_main():
    sim_time = 0
    step = 0.0001
    while True:
        sim_time += step
        simulator.AdvanceTo(sim_time)

if __name__ == '__main__':
    if enable_camera:
        camera_main()
    else:
        simulator.AdvanceTo(0.01)