import argparse
import ruamel.yaml as yaml
import json
import os

import utils
from dataset import create_dataset
from models import create_model


class AndroidEnv:
    def __init__(self, 
        avd_name, 
        cache_avd_names,
        udids,
        appium_base_port,
        android_avd_home: str = '/nfs/kun2/users/yifei/openended/.android/android_avd/avd',
        emulator_path: str = '~/Android/Sdk/emulator/emulator',
        adb_path: str = "~/Library/Android/sdk/platform-tools/adb",
        run_headless: bool = False,
        max_steps: int = 10,
        evaluators = None,
        prepare_prompt = autoui_prepare_prompt, 
        temp_path = "/nfs/kun2/users/yifei/openended/logs/images",
        save_images = False,
        all_tasks = None,
        task_split = "train",
        sample_mode = None,
        record = False
    ):
        
        self.android_avd_home = os.path.expanduser(android_avd_home)
        self.emulator_path = os.path.expanduser(emulator_path)
        self.adb_path = os.path.expanduser(adb_path)
        self.avd_name = avd_name
        self.save_images = save_images
        self.bsize = len(cache_avd_names)
        self.cache_avd_names = cache_avd_names
        self.run_headless = run_headless
        self.max_steps = max_steps
        self.emulator_group_offset = 0
        self.record = record
        self.all_tasks = all_tasks
        self.task_split = task_split
        self.prepare_prompt = prepare_prompt
        self.translate_action = translate_action
        self.temp_path = temp_path
        self.evaluators = evaluators
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        self.udids = udids
        self.base_port = appium_base_port
        self.appium_processes = []
        self.sample_mode = sample_mode

        # Start the appium servers
        for i in range(self.base_port, self.base_port+self.bsize):
            self.appium_processes.append(subprocess.Popen(f"appium --relaxed-security -p {i} > /dev/null", stdout=subprocess.DEVNULL, shell=True))
            print("starting appium server at port ", i)
        self.appium_server_urls = [f"http://localhost:{i}" for i in range(self.base_port, self.base_port+self.bsize)]


    def reset(self):
        """
        Reset the emulator to a clean state
        """
        # If the emulator is already running, kill it,
        # Then delete the cache AVD
        kill_all_emulators(self.adb_path, emulators=self.udids)
        if hasattr(self, "emulator_process"):
            self.emulator_process.send_signal(signal.SIGINT)
            self.emulator_process.wait()
        self.emulators = []
        for cache_avd_name in self.cache_avd_names:
            # print(cache_avd_name)
            for _ in range(3):
                try:
                    cache_avd_path = os.path.join(self.android_avd_home, cache_avd_name + ".avd")
                    cache_avd_ini_path = os.path.join(self.android_avd_home, cache_avd_name + ".ini")
                    if os.path.exists(cache_avd_path):
                        shutil.rmtree(cache_avd_path, ignore_errors=True)
                    if os.path.exists(cache_avd_ini_path):
                        os.remove(cache_avd_ini_path)
                    sleep(2)
                    # Clone the source AVD and start the emulator
                    clone_avd(self.avd_name, cache_avd_name, self.android_avd_home)
                    break
                except OSError as e:
                    print(f"Failed to reset the emulator: {e}")
                    import traceback
                    print(traceback.format_exc())
                    sleep(20)


        def emulator_constructor(udid, appium_server_url, cache_avd_name, evaluator, task_id, task_split):
            return AndroidEmulator(avd_name=cache_avd_name, max_steps=self.max_steps, emulator_path=self.emulator_path, 
                appium_server_url=appium_server_url, 
                no_window=self.run_headless, 
                udid = udid,
                feature_extractor = self.feature_extractor,
                prepare_prompt = self.prepare_prompt,
                translate_action = self.translate_action,
                all_tasks = self.all_tasks,
                evaluator = evaluator,
                temp_path = os.path.join(self.temp_path, cache_avd_name),
                save_images = self.save_images,
                task_id=task_id,
                task_split=task_split,
                sample_mode=self.sample_mode,
                record=self.record)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(emulator_constructor, udid, appium_server_url, cache_avd_name, evaluator, task_id, self.task_split)
                for udid, appium_server_url, cache_avd_name, evaluator, task_id in 
                zip(self.udids, self.appium_server_urls, self.cache_avd_names, self.evaluators, range(self.emulator_group_offset, self.emulator_group_offset+self.bsize))]
            self.emulators = [job.result() for job in jobs]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(emulator.get_obs) for emulator in self.emulators]
            # for i, job in enumerate(jobs):
                # colorful_print(f"Getting observation from emulator {i}: {job.result()}", "green")
            return [job.result() for job in jobs]

    def step(self, actions):
        if not self.emulators:
            raise Exception("Please call reset() before calling step()")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(emulator.step, action) 
                    for emulator, action in 
                    zip(self.emulators, actions)]
            results = [job.result() for job in jobs]
        return results



def evaluation(config, model, dataloader):
    pass


def main(args, config):
    print("### Evaluating")

    print("config:", json.dumps(config))
    output_path = os.path.join("checkpoints/results/", f"{config['model_name']}.jsonl")
    print("output_path: ", output_path)

    print("### Creating model")
    model = create_model(config)

    model.eval()

    print("### Total Params: ", sum(p.numel() for p in model.parameters()))

    print("### Creating datasets")
    dataloader = create_dataset('eval', config)

    print("### Start evaluating")

    predictions = evaluation(config, model, dataloader)

    utils.write_jsonl(predictions, output_path)
    print("### Prediction Results Save To: ", output_path)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, default="")
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    main(args, config)