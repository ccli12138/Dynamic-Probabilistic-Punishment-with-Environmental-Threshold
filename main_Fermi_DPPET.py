import torch
from Fermi_DPPET import SPGG
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import asyncio
import argparse
import os
import shutil
import re
def save_params_to_json(params, filename_prefix="params",output_path='data'):
    # 创建参数保存目录
    param_dir = Path(output_path)
    os.makedirs(output_path, exist_ok=True)
    # param_dir.mkdir(exist_ok=True)
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"
    filepath = param_dir / filename
    
    # 转换特殊数据类型
    serializable_params = {
        k: str(v) if isinstance(v, torch.device) else v  # 处理device类型
        for k, v in params.items()
    }
    
    # 保存到JSON
    with open(filepath, 'w') as f:
        json.dump(serializable_params, f, indent=4)
    
    src_file='main_Fermi_cc_v4.py'
    dst_file=f'{output_path}/{src_file}'
    shutil.copy2(src_file, dst_file)
    src_file='Fermi_cc_v4.py'
    dst_file=f'{output_path}/{src_file}'
    shutil.copy2(src_file, dst_file)
    
    print(f"参数已保存至: {filepath}")

# 主实验程序
async def main(args):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H%M%S")
    # 实验参数设置
    fontsize=16
    r_values =[2.0]#[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]#[4.5,4.6,4.7,4.8,4.9,5.0,5.1]#[3.6, 3.8, 4.7, 5.0, 5.5, 6.0] #[3.0, 5.0, 7.0, 9.0]  # 公共物品乘数
    # 使用 arange 生成从 1 到 6 的列表，间隔为 0.1
    # r_values = [round(i * 0.1, 1) for i in range(45, 56)]
    # r_values = [round(i * 0.1, 1) for i in range(30, 41)]
    # print(result_list)

    if args.device=='cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device=='cpu':
        device = torch.device("cpu")
    elif args.device=='mps':
        device = torch.device("mps")

    if args.epochs==100000:
        xticks=[1, 10, 100, 1000, 10000, 100000]
    elif args.epochs==10000:
        xticks=[1, 10, 100, 1000, 10000]
    elif args.epochs==1000:
        xticks=[1, 10, 100, 1000]
    # fra_yticks=[0.00, 0.20, 0.40, 0.60, 0.80, 1.00]
    fra_yticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    profite_yticks=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # 实验参数设置
    experiment_params = {
        "r": r_values,
        "epochs": args.epochs,
        "runs": args.runs,
        "L_num": args.L_num,
        "alpha": args.alpha,
        "lam": args.lam,
        "delta": args.delta,
        "beta": args.beta,
        "is_cc": args.is_cc,
        "xticks": xticks,
        "fra_yticks": fra_yticks,
        "profite_yticks": profite_yticks,
        "start_time": formatted_time
    }

    if args.is_cc:
        output_path = f'data/Fermi_cc_v4_{formatted_time}_q{args.question}_e{args.epochs}_L{args.L_num}_seed{args.seed}_alpha{args.alpha}_lam{args.lam}_delta{args.delta}_beta{args.beta}'
    else:
        output_path = f'data/Fermi_v4_{formatted_time}_q{args.question}_e{args.epochs}_L{args.L_num}_seed{args.seed}_alpha{args.alpha}_lam{args.lam}_delta{args.delta}_beta{args.beta}'

    save_params_to_json(experiment_params, filename_prefix="params",output_path=output_path)
    
    C_Y_list=[]
    P_Y_list=[]
    # 实验循环
    for r in r_values:
        print(f"\nRunning experiment with r={r}")
        # 多次独立运行
        for num in range(args.runs):
            # 初始化模型
            model = SPGG(
                L_num=args.L_num,
                device=device,
                r=r,
                epoches=args.epochs,
                now_time=formatted_time,
                question=args.question,
                is_cc=args.is_cc,
                output_path=output_path,
                alpha=args.alpha,
                lam=args.lam,
                delta=args.delta,
                beta=args.beta
                )
            print(f"Run {num+1}/{args.runs}")
            model.count = num  # 记录运行次数
            
            # 执行模拟
            D_Y, C_Y, P_Y, all_value = model.run(num=num)
            C_Y_list.append(C_Y)
            P_Y_list.append(P_Y)

            
            # 保存实验结果
            model.save_data('Density_D', f'Density_D_r{r}', r, num, D_Y)
            model.save_data('Density_C', f'Density_C_r{r}', r, num, C_Y)
            model.save_data('Density_P', f'Density_P_r{r}', r, num, P_Y)
            model.save_data('Total_Value', f'Total_Value_r{r}', r, num, all_value)

            plt.clf()
            plt.close("all")

            # plt.xticks(xticks, [str(x) for x in xticks], fontsize=fontsize)  # 强制显示预设刻度
            plt.yscale('linear')  # 保持y轴线性尺度（更直观观察比例变化）
            plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)  # 仅显示y轴网格
            plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=1)  # 添加50%参考线
            # 修改后的绘图部分（替换原有合作演化绘图代码）
            # 绘制策略演化图
            plt.figure(figsize=(8, 6))

            plt.xscale('symlog', 
                linthresh=1,      # 线性/对数分界
                linscale=0.5,     # 线性区域压缩程度
                subs=np.arange(1,10))      # 对数区间的次要刻度


            plt.plot( C_Y, 'b-', linewidth=2, alpha=0.7, label='C')
            plt.plot( D_Y, 'r-', linewidth=2, alpha=0.7, label='D')
            plt.plot( P_Y, 'g-', linewidth=2, alpha=0.7, label='P')
            
            plt.xticks(xticks, [str(x) for x in xticks], fontsize=fontsize)
            plt.yticks(fra_yticks, fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xlim(0, None)
            
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.xlabel('t', fontsize=fontsize, labelpad=10)
            plt.ylabel('Fractions', fontsize=fontsize, labelpad=10)
            plt.legend(loc='best', fontsize=fontsize)
            
            # plt.savefig(f'{output_path}/strategy_evolution_r{r}_run{num}.png', 
            #            dpi=300, bbox_inches='tight', pad_inches=0)
            plt.savefig(f'{output_path}/strategy_evolution_r{r}_run{num}.pdf', 
                       format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)

            plt.close()

    # 收集所有r值对应的最终策略比例
    r_list = []
    c_final = []
    d_final = []
    p_final = []
    
    # 遍历所有结果文件
    for r in r_values:
        try:
            # 读取合作者比例
            c_file = f'{output_path}/Density_C/Density_C_r{r}_run0.txt'
            with open(c_file, 'r') as f:
                c_data = f.readlines()
                c_final.append(float(c_data[-1]))
            
            # 读取叛逃者比例
            d_file = f'{output_path}/Density_D/Density_D_r{r}_run0.txt'
            with open(d_file, 'r') as f:
                d_data = f.readlines()
                d_final.append(float(d_data[-1]))
            
            # 读取惩罚者比例
            p_file = f'{output_path}/Density_P/Density_P_r{r}_run0.txt'
            with open(p_file, 'r') as f:
                p_data = f.readlines()
                p_final.append(float(p_data[-1]))
            
            r_list.append(r)
        except FileNotFoundError:
            continue
    
    # 绘制图形
    plt.figure(figsize=(8, 6))
    plt.plot(r_list, c_final, 'bo-', label='C')
    plt.plot(r_list, d_final, 'rd-', label='D')
    plt.plot(r_list, p_final, 'gs-', label='P')
    
    plt.xlabel('r', fontsize=fontsize)
    plt.ylabel('Fraction', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # plt.ylim(0, 1)
    plt.grid(True)
    plt.legend(loc='best',fontsize=fontsize)
    
    # plt.savefig(f'{output_path}/final_strategy_ratios.png', 
    #         dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig(f'{output_path}/final_strategy_ratios.pdf', 
            format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 绘制不同r下的C+P在t时间的演化（画在同一张图上，legend是r），合作者和惩罚者的比例要加在一起，使用C_Y_list和P_Y_list的数据
    plt.figure(figsize=(8, 6))
    for i, r in enumerate(r_values):
        now_C=np.array(C_Y_list[i])
        now_P=np.array(P_Y_list[i])
        plt.plot(now_C+now_P, label=f'r={r}')
        
    plt.xlabel('t', fontsize=fontsize)
    plt.ylabel('$f_c$', fontsize=fontsize)
    plt.xticks(xticks, [str(x) for x in xticks], fontsize=fontsize)
    plt.yticks(fra_yticks, fontsize=fontsize)
    # plt.ylim(0, 1)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=fontsize)
    plt.xlim(0, None)
    plt.xscale('symlog', 
           linthresh=1,      # 线性/对数分界
           linscale=0.5,     # 线性区域压缩程度
           subs=np.arange(1,10))      # 对数区间的次要刻度

    
    plt.savefig(f'{output_path}/CP_evolution_by_r.pdf', 
                    format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig

    


    print("All experiments completed!")

if __name__ == "__main__":
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description='Process some parameters.')

    # 添加args参数
    # parser.add_argument('-r', type=float, default=0.5, help='R parameter')
    parser.add_argument('-epochs', type=int, default=10000, help='Epochs')
    parser.add_argument('-runs', type=int, default=1, help='Runs')
    parser.add_argument('-L_num', type=int, default=200, help='question size')
    parser.add_argument('-question', type=int, default=1, help='question')
    parser.add_argument('-alpha', type=float, default=0.5, help='Punishment cost')
    parser.add_argument('-lam', type=float, default=0.2, help='Uncertainty in punishment probability')
    parser.add_argument('-delta', type=float, default=0.5, help='Social norm threshold')
    parser.add_argument('-beta', type=float, default=1.0, help='Punishment fine')
    parser.add_argument('-device', type=str, default='cuda', help='Device')
    parser.add_argument('-is_cc', action='store_true', default=True, help='is cc')
    parser.add_argument('-is_not_cc', action='store_false', dest='is_cc', help='is not cc')
    parser.add_argument('-seed', type=int, default=1, help='random seed')

    # 解析命令行参数
    args = parser.parse_args()

    # AC_P:  r4.8:28,31,37,38,44,61,62,63
    # seed=2 # r5.0: 3,6,8,9,10, 
    # 固定 numpy 的随机数种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # args.seed=seed
    asyncio.run(main(args))