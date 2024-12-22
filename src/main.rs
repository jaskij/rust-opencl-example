use anyhow::Result;
use log::info;
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::event::cl_event;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::CL_NON_BLOCKING;
use std::fs::File;
use std::io::BufWriter;
use std::ptr;
use std::time::Duration;

const KERNEL_SOURCE: &str = include_str!("test.cl");
const KERNEL_NAME: &str = "mandelbrot";

const IMAGE_WIDTH: usize = 640;
const IMAGE_HEIGHT: usize = 480;
const IMAGE_SIZE: usize = IMAGE_WIDTH * IMAGE_HEIGHT;
const IMAGE_ARRAY_SIZE: usize = IMAGE_SIZE * 3;

fn main() -> Result<()> {
    env_logger::init();

    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
        .first()
        .expect("no devices found");
    let device = Device::new(device_id);
    info!("running on {} {} ", device.vendor()?, device.name()?);

    let runtime_start = std::time::Instant::now();

    let context = Context::from_device(&device)?;
    let program = Program::create_and_build_from_source(&context, KERNEL_SOURCE, "")
        .expect("failed to compile kernel");
    let kernel = Kernel::create(&program, KERNEL_NAME)?;

    let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)?;

    let buffer = unsafe {
        Buffer::<u8>::create(
            &context,
            CL_MEM_WRITE_ONLY,
            IMAGE_ARRAY_SIZE,
            ptr::null_mut(),
        )?
    };

    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&buffer)
            .set_global_work_sizes(&[IMAGE_WIDTH, IMAGE_HEIGHT])
            .enqueue_nd_range(&queue)?
    };
    let events: Vec<cl_event> = vec![kernel_event.get()];

    let mut results = vec![0; IMAGE_ARRAY_SIZE];
    let read_event =
        unsafe { queue.enqueue_read_buffer(&buffer, CL_NON_BLOCKING, 0, &mut results, &events)? };
    read_event.wait()?;

    let run_duration = runtime_start.elapsed();

    let file = File::create("out.png")?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, IMAGE_WIDTH as u32, IMAGE_HEIGHT as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);

    let mut writer = encoder.write_header()?;
    writer.write_image_data(&results)?;

    let start_time = kernel_event.profiling_command_start()?;
    let end_time = kernel_event.profiling_command_end()?;
    let kernel_runtime = Duration::from_nanos(end_time - start_time);

    info!(
        "program took {:?}, of which kernel took {:?}",
        run_duration, kernel_runtime
    );

    Ok(())
}
