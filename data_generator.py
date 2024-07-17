import numpy as np
import pywt
import matplotlib.pyplot as plt
import japanize_matplotlib
from matplotlib.widgets import Slider, RadioButtons
import pywt
from sklearn.metrics import mean_squared_error


def wavelet_denoise(signal, wavelet, mode, threshold):
    coeffs = pywt.wavedec(signal, wavelet)
    coeffs_thresh = [pywt.threshold(c, threshold, mode=mode) for c in coeffs]
    return pywt.waverec(coeffs_thresh, wavelet)


def main():
    # 初期パラメータ
    wavelet = 'haar'
    initial_threshold = 0.0
    initial_mode = 'hard'
    initial_func_type = 0

    # データ列の作成
    np.random.seed(0)
    time = np.linspace(0, 1, 10000)
    original_signal = np.sin(2 * np.pi * time)
    noisy_signal = original_signal + \
        np.random.normal(0, 0.2, time.shape)

    # ウェーブレット変換
    denoised_signal = wavelet_denoise(
        noisy_signal, wavelet, initial_mode, initial_threshold)
    initial_mse = mean_squared_error(original_signal, denoised_signal)
    initial_rmse = np.sqrt(initial_mse)

    # プロットの準備
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.3, bottom=0.3)
    l_denoised, = ax.plot(time, denoised_signal, label='Denoised Signal')
    l_original, = ax.plot(time, original_signal, label='Original Signal')
    ax.legend()
    ax.set_title('Wavelet Denoise Tuner')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')

    # コンポーネントの位置
    threshold_slider_pos = plt.axes([0.1, 0.17, 0.8, 0.03],
                                    facecolor='lightgoldenrodyellow')
    mode_radiobutton_pos = plt.axes([0.05, 0.4, 0.15, 0.1],
                                    facecolor='lightgoldenrodyellow')
    function_radiobutton_pos = plt.axes([0.05, 0.7, 0.15, 0.2],
                                        facecolor='lightgoldenrodyellow')

    # スライダーの追加
    slider_threshold = Slider(threshold_slider_pos, 'Threshold',
                              0.0, 1.0, valinit=initial_threshold)

    # モード選択用ラジオボタンの追加
    radio_mode = RadioButtons(mode_radiobutton_pos, ('soft', 'hard'), active=0)

    # 関数選択用ラジオボタンの追加
    radio_func = RadioButtons(function_radiobutton_pos,
                              ('関数1', '関数2', '関数3', '関数4'), active=0)

    # rmseを表示するテキストの追加
    rmse_text = fig.text(
        0.5, 0.08, f'RMSE（バラバラ度）：{initial_rmse:.3f}', va='center', ha='center', fontsize=30, color='blue')

    # 更新関数
    def update(val):
        # 解析対象関数の再定義
        func_type = radio_func.value_selected
        if func_type == '関数1':
            np.random.seed(0)
            original_signal = np.sin(2 * np.pi * time)
            noisy_signal = original_signal + \
                np.random.normal(0, 0.2, time.shape)
        elif func_type == '関数2':
            np.random.seed(0)
            original_signal = 0.5 * np.sin(2 * np.pi * time) + 0.25 * np.tan(2 * np.pi * time)
            noisy_signal = original_signal + \
                np.random.normal(0, 0.2, time.shape)
        elif func_type == '関数3':
            np.random.seed(0)
            original_signal = np.sin(2 * np.pi * time) + \
                np.sin(4 * np.pi * time)
            noisy_signal = original_signal + \
                np.random.normal(0, 0.5, time.shape)
        elif func_type == '関数4':
            np.random.seed(0)
            original_signal = np.sin(2 * np.pi * time) + \
                np.sin(4 * np.pi * time)
            noisy_signal = original_signal + \
                np.random.normal(0, 0.8, time.shape)

        # ウェーブレット変換
        threshold = slider_threshold.val
        mode = radio_mode.value_selected
        denoised_signal = wavelet_denoise(
            noisy_signal, wavelet, mode, threshold)
        l_denoised.set_ydata(denoised_signal)
        l_original.set_ydata(original_signal)

        # RMSEの計算
        mse = mean_squared_error(original_signal, denoised_signal)
        rmse = np.sqrt(mse)
        rmse_text.set_text(f'RMSE（バラバラ度）：{rmse:.3f}')

        fig.canvas.draw_idle()

    # コンポーネントの更新時update関数を実行
    slider_threshold.on_changed(update)
    radio_mode.on_clicked(update)
    radio_func.on_clicked(update)

    plt.show()


if __name__ == '__main__':
    main()
