use num::Complex;
use std::{borrow::Cow, str::FromStr};
use wgpu::util::DeviceExt;
use wgpu::*;

pub struct GPUHandle {
    device: Device,
    queue: Queue,
    staging_buffer: Buffer,
    storage_buffer: Buffer,
    prng_buffer: Buffer,
    gpu_vars_buffer: Buffer,
    compute_pipeline: ComputePipeline,
    bind_group: BindGroup,
    width: u32,
    height: u32,
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
struct GPUVars {
    width: u32,
    height: u32,
    max_iterations: u32,
    ll_re: f32,
    ll_im: f32,
    ur_re: f32,
    ur_im: f32,
    zoom_ll_re: f32,
    zoom_ll_im: f32,
    zoom_ur_re: f32,
    zoom_ur_im: f32,
}

impl GPUHandle {
    pub fn new(trialsx2: u32, width: u32, height: u32) -> Self {
        assert!(trialsx2 % 6400 == 0);

        let (device, queue) = pollster::block_on(GPUHandle::initialize());

        let (
            staging_buffer,
            storage_buffer,
            prng_buffer,
            gpu_vars_buffer,
            compute_pipeline,
            bind_group,
        ) = pollster::block_on(GPUHandle::setup_compute(&device, trialsx2, width, height));

        GPUHandle {
            device,
            queue,
            staging_buffer,
            storage_buffer,
            gpu_vars_buffer,
            prng_buffer,
            compute_pipeline,
            bind_group,
            width,
            height,
        }
    }

    async fn setup_compute(
        device: &Device,
        trialsx2: u32,
        width: u32,
        height: u32,
    ) -> (Buffer, Buffer, Buffer, Buffer, ComputePipeline, BindGroup) {
        // Loads the shader from WGSL
        let cs_module = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: (std::mem::size_of::<u32>() as u32 * width * height) as BufferAddress,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let storage_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Storage Buffer"),
            size: (std::mem::size_of::<u32>() as u32 * width * height) as BufferAddress,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let prng_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("PRNG Buffer"),
            size: std::mem::size_of::<f32>() as BufferAddress * trialsx2 as BufferAddress,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let gpu_vars_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Variables Buffer"),
            size: std::mem::size_of::<GPUVars>() as BufferAddress,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // A bind group defines how buffers are accessed by shaders.
        // It is to WebGPU what a descriptor set is to Vulkan.
        // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).

        // A pipeline specifies the operation of a shader

        // Instantiates the pipeline.
        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &cs_module,
            entry_point: "main",
        });

        // Instantiates the bind group, once again specifying the binding of buffers.
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: storage_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: prng_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: gpu_vars_buffer.as_entire_binding(),
                },
            ],
        });

        (
            staging_buffer,
            storage_buffer,
            prng_buffer,
            gpu_vars_buffer,
            compute_pipeline,
            bind_group,
        )
    }

    #[cfg_attr(test, allow(dead_code))]
    async fn initialize() -> (Device, Queue) {
        // Instantiates instance of WebGPU
        let instance = Instance::default();

        // `request_adapter` instantiates the general connection to the GPU
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .await
            .unwrap();

        // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
        //  `features` being the available features.

        let required_limits = {
            let mut x = Limits::downlevel_defaults();
            x.max_buffer_size = 2000000000;
            x.max_storage_buffer_binding_size = 2000000000;
            x.max_uniform_buffer_binding_size = 2000000000;
            x
        };

        adapter
            .request_device(
                &DeviceDescriptor {
                    label: None,
                    required_features: Features::empty(),
                    required_limits,
                },
                None,
            )
            .await
            .unwrap()
    }

    async fn execute_gpu(
        &mut self,
        ll: Complex<f32>,
        ur: Complex<f32>,
        zoom_ll: Complex<f32>,
        zoom_ur: Complex<f32>,
        max_iterations: u32,
        prng_data: Vec<f32>,
    ) -> Option<Vec<u32>> {
        let gpu_vars = GPUVars {
            width: self.width,
            height: self.height,
            max_iterations,
            ll_re: ll.re,
            ll_im: ll.im,
            ur_re: ur.re,
            ur_im: ur.im,
            zoom_ll_re: zoom_ll.re,
            zoom_ll_im: zoom_ll.im,
            zoom_ur_re: zoom_ur.re,
            zoom_ur_im: zoom_ur.im,
        };

        let zero_data = {
            let mut v = vec![];
            for _ in 0..self.storage_buffer.size() {
                v.push(0);
            }
            v
        };

        self.queue.write_buffer(&self.storage_buffer, 0, &zero_data);

        self.queue
            .write_buffer(&self.gpu_vars_buffer, 0, bytemuck::bytes_of(&gpu_vars));

        self.queue
            .write_buffer(&self.prng_buffer, 0, bytemuck::cast_slice(&prng_data));

        // A command encoder executes one or many pipelines.
        // It is to WebGPU what a command buffer is to Vulkan.
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.insert_debug_marker("compute buddhabrot iterations");
            cpass.dispatch_workgroups(100, prng_data.len() as u32 / 64 / 100 / 2, 1);
            // 6400/2 = 64 * 100 * (6400 /64/100/2)
        }
        // Sets adds copy operation to command encoder.
        // Will copy data from storage buffer on GPU to staging buffer on CPU.
        encoder.copy_buffer_to_buffer(
            &self.storage_buffer,
            0,
            &self.staging_buffer,
            0,
            self.storage_buffer.size(),
        );

        // Submits command encoder for processing
        self.queue.submit(Some(encoder.finish()));

        // Note that we're not calling `.await` here.
        let buffer_slice = self.staging_buffer.slice(..);
        // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(MapMode::Read, move |v| sender.send(v).unwrap());

        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        self.device.poll(Maintain::wait()).panic_on_timeout();

        // Awaits until `buffer_future` can be read from
        if let Ok(Ok(())) = receiver.recv_async().await {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            let result = bytemuck::cast_slice(&data).to_vec();

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            self.staging_buffer.unmap(); // Unmaps buffer from memory
                                         // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                         //   delete myPointer;
                                         //   myPointer = NULL;
                                         // It effectively frees the memory

            // Returns data from buffer
            Some(result)
        } else {
            panic!("failed to run compute on gpu!")
        }
    }

    pub fn call(
        &mut self,
        ll: Complex<f32>,
        ur: Complex<f32>,
        zoom_ll: Complex<f32>,
        zoom_ur: Complex<f32>,
        max_iterations: u32,
        prng_data: Vec<f32>,
    ) -> Vec<u32> {
        pollster::block_on(self.execute_gpu(ll, ur, zoom_ll, zoom_ur, max_iterations, prng_data))
            .unwrap()
    }
}
