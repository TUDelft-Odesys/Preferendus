import pathlib

from pyFAST.case_generation import runner

HERE = pathlib.Path(__file__).parent
print(f'I am here: {HERE}')

speed = 10
max_speed = 35
waiter = 0

while speed <= max_speed:
    with open(f'{HERE}\90m_base.inp', 'r') as t:
        text = t.read()

    text = text.replace('xxxx', str(speed))

    with open(f'{HERE}\90m_{speed}mps.inp', 'w') as file:
        file.writelines(text)

    if waiter < 3 and speed != max_speed:
        print(f'Running wind speed {speed} without waiting')
        run = runner.run_cmd(f'{HERE}\90m_{speed}mps.inp', f'{HERE}\TurbSim_x64.exe', wait=False)
        waiter += 1
    else:
        print(f'Running wind speed {speed} with waiting')
        runner.run_cmd(f'{HERE}\90m_{speed}mps.inp', f'{HERE}\TurbSim_x64.exe', wait=True)
        waiter = 0

    speed += 0.5
