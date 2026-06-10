document.addEventListener("DOMContentLoaded", () => {
    // 0. DOM Element Declarations
    const langEnBtn = document.getElementById("lang-en-btn");
    const langRuBtn = document.getElementById("lang-ru-btn");
    const generateBtn = document.getElementById("generate-btn");
    const statusBox = document.getElementById("status-box");
    const statusText = document.getElementById("status-text");
    const statusPercent = document.getElementById("status-percent");
    const progressBar = document.getElementById("progress-bar");
    const logOutput = document.getElementById("log-output");
    
    const modelViewer = document.getElementById("model-viewer");
    const viewportPlaceholder = document.getElementById("viewport-placeholder");
    const maskPreview = document.getElementById("mask-preview");
    const maskPlaceholder = document.getElementById("mask-placeholder");
    const renderPreview = document.getElementById("render-preview");
    const renderPlaceholder = document.getElementById("render-placeholder");
    const downloadsTabBtn = document.getElementById("downloads-tab-btn");
    const filesList = document.getElementById("files-list");
    
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const dropzoneEmpty = document.getElementById("dropzone-empty");
    const dropzonePreview = document.getElementById("dropzone-preview");
    const imagePreview = document.getElementById("image-preview");
    const removeImgBtn = document.getElementById("remove-img-btn");

    // 1. Bilingual Translations Dictionary
    const translations = {
        en: {
            header_title: "FormaAi Studio",
            header_subtitle: "Unified Hybrid 3D Generation Pipeline (Mesh + Gaussians)",
            input_image: "1. Input Image",
            upload_text: "Drag & drop image here or ",
            upload_browse: "choose file",
            upload_hint: "Supports PNG, JPG, JPEG",
            remove_btn: "Delete",
            gen_params: "2. Generation Settings",
            seed_label: "Random Seed",
            stage1_steps: "Sparse Structure Steps (Stage 1):",
            stage1_cfg: "Sparse Structure CFG (Guidance):",
            stage2_steps: "Structured Latent Steps (Stage 2):",
            stage2_cfg: "Structured Latent CFG (Guidance):",
            postprocess_title: "3. Postprocessing & AI Upscaling",
            refine_gs_label: "Refine 3D Gaussians (Differentiable Refinement)",
            refine_steps_label: "Optimization Steps:",
            mesh_upscaling: "Mesh Texture Scaling (AI Super Resolution)",
            upscale_off: "Off",
            upscale_device: "Upscaling Device Selection",
            device_gpu: "GPU (Fast)",
            device_cpu: "CPU (No OOM)",
            device_auto: "Auto-Fallback",
            generate_btn: "Generate 3D Object",
            status_label: "Status:",
            result_title: "4. Generation Result",
            viewport_placeholder: "GLB 3D model will appear here after generation completes",
            tab_preview: "Add. Preview",
            tab_downloads: "Downloads",
            mask_title: "Image Mask (Rembg)",
            render_title: "3D Gaussians Render (Front View)",
            mask_placeholder: "Waiting for generation...",
            render_placeholder: "Waiting for generation...",
            downloads_empty: "Result files not generated.",
            
            // Dynamic text
            alert_select_image: "Please select an image file.",
            alert_upload_image: "Please upload an input image in section 1 first.",
            log_submitting: "Initializing form submission...",
            btn_generating: "Generating...",
            log_registered: "Task registered. ID: ",
            log_failed_submit: "Critical submit failure: ",
            err_polling: "Failed to poll server status.",
            log_success: "Pipeline completed successfully!",
            log_failed: "Generation failed: ",
            log_poll_error: "Polling error: ",
            unknown_error: "Unknown error",
            
            status_pending: "In Queue...",
            status_processing: "Generating...",
            status_completed: "Completed",
            status_failed: "Failed",
            
            file_glb_desc: "3D Mesh (GLB)",
            file_obj_desc: "Textured OBJ Mesh Package (ZIP Archive)",
            file_ply_desc: "Point Cloud PLY",
            file_json_desc: "Alignment Metadata (JSON)",
            btn_download: "Download"
        },
        ru: {
            header_title: "FormaAi Studio",
            header_subtitle: "Полностью асинхронный гибридный пайплайн 3D-генерации (Mesh + Gaussians)",
            input_image: "1. Входные данные",
            upload_text: "Перетащите изображение сюда или ",
            upload_browse: "выберите файл",
            upload_hint: "Поддерживаются PNG, JPG, JPEG",
            remove_btn: "Удалить",
            gen_params: "2. Параметры генерации",
            seed_label: "Случайное зерно (Seed)",
            stage1_steps: "Sparse Structure Steps (Шаги Stage 1):",
            stage1_cfg: "Sparse Structure CFG (Guidance):",
            stage2_steps: "Structured Latent Steps (Шаги Stage 2):",
            stage2_cfg: "Structured Latent CFG (Guidance):",
            postprocess_title: "3. Постобработка и AI Апскейлинг",
            refine_gs_label: "Уточнение 3D Gaussians (Differentiable Refinement)",
            refine_steps_label: "Шаги оптимизации:",
            mesh_upscaling: "Масштабирование текстуры меша (AI Super Resolution)",
            upscale_off: "Выкл",
            upscale_device: "Выбор устройства для апскейлинга",
            device_gpu: "GPU (Быстро)",
            device_cpu: "CPU (Без OOM)",
            device_auto: "Auto-Fallback",
            generate_btn: "Сгенерировать 3D объект",
            status_label: "Статус:",
            result_title: "4. Результат генерации",
            viewport_placeholder: "3D-модель GLB отобразится здесь после завершения генерации",
            tab_preview: "Доп. Превью",
            tab_downloads: "Файлы для скачивания",
            mask_title: "Маска изображения (Rembg)",
            render_title: "Рендер 3D Gaussians (Передний ракурс)",
            mask_placeholder: "Ожидание генерации...",
            render_placeholder: "Ожидание генерации...",
            downloads_empty: "Файлы результатов не сгенерированы.",
            
            // Dynamic text
            alert_select_image: "Пожалуйста, выберите файл изображения.",
            alert_upload_image: "Пожалуйста, загрузите исходное изображение в зону 1.",
            log_submitting: "Инициализация отправки формы...",
            btn_generating: "Генерируется...",
            log_registered: "Задача зарегистрирована. ID: ",
            log_failed_submit: "Критический сбой: ",
            err_polling: "Не удалось опросить сервер.",
            log_success: "Успешное завершение пайплайна!",
            log_failed: "Сбой генерации: ",
            log_poll_error: "Ошибка опроса: ",
            unknown_error: "Неизвестная ошибка",
            
            status_pending: "В очереди...",
            status_processing: "Генерация...",
            status_completed: "Готово",
            status_failed: "Ошибка",
            
            file_glb_desc: "3D Меш (GLB)",
            file_obj_desc: "Текстурированный меш OBJ (ZIP-архив)",
            file_ply_desc: "Облако точек PLY",
            file_json_desc: "Метаданные выравнивания (JSON)",
            btn_download: "Скачать"
        }
    };

    // Stage translation dictionary for English logs
    const stageTranslations = {
        "Инициализация задачи в очереди...": "Initializing task in queue...",
        "Вырезание фона и центрирование (Rembg)...": "Removing background & centering (Rembg)...",
        "Запуск 3D-генерации (Stage 1 & Stage 2)...": "Running 3D generation (Stage 1 & Stage 2)...",
        "Рендеринг превью 3D-гауссианов...": "Rendering 3D Gaussians preview...",
        "Экспорт полигональной сетки и запекание текстур (GLB)...": "Exporting mesh & baking textures (GLB)...",
        "Генерация успешно завершена!": "3D asset generation completed successfully!",
        "Ошибка генерации": "Generation failed."
    };

    // 2. Setup Language Switcher State & Toggle Event Listeners
    let currentLang = localStorage.getItem("forma_lang") || "en";
    let lastResults = null;

    function updateLanguageUI(lang) {
        currentLang = lang;
        localStorage.setItem("forma_lang", lang);

        if (lang === "en") {
            langEnBtn.classList.add("active");
            langRuBtn.classList.remove("active");
        } else {
            langRuBtn.classList.add("active");
            langEnBtn.classList.remove("active");
        }

        // Translate all static elements with data-i18n
        document.querySelectorAll("[data-i18n]").forEach(el => {
            const key = el.getAttribute("data-i18n");
            if (translations[lang] && translations[lang][key]) {
                if (key === "generate_btn") {
                    if (!generateBtn.disabled) {
                        el.textContent = translations[lang][key];
                    } else {
                        el.textContent = translations[lang].btn_generating;
                    }
                } else {
                    el.textContent = translations[lang][key];
                }
            }
        });

        // Translate dynamic elements if currently generating/complete
        if (statusBox.style.display !== "none") {
            const currentStatus = statusText.getAttribute("data-status") || "pending";
            statusText.textContent = translations[lang]["status_" + currentStatus];
        }

        if (lastResults) {
            loadResults(lastResults);
        }
    }

    langEnBtn.addEventListener("click", () => updateLanguageUI("en"));
    langRuBtn.addEventListener("click", () => updateLanguageUI("ru"));

    // Initial load language setup
    updateLanguageUI(currentLang);

    // 3. Accordion Toggle Logic
    setupAccordion("accordion-basic-btn", "accordion-basic-content");
    setupAccordion("accordion-post-btn", "accordion-post-content");
    
    // Automatically open accordion contents initially
    document.getElementById("accordion-basic-btn").parentElement.classList.add("open");
    document.getElementById("accordion-post-btn").parentElement.classList.add("open");
    recalculateAccordionHeight("accordion-basic-content");
    recalculateAccordionHeight("accordion-post-content");

    // 4. Sliders Input Update Logic
    setupSliderBadge("ss-steps-input", "ss-steps-val");
    setupSliderBadge("ss-cfg-input", "ss-cfg-val");
    setupSliderBadge("slat-steps-input", "slat-steps-val");
    setupSliderBadge("slat-cfg-input", "slat-cfg-val");
    setupSliderBadge("refine-steps-input", "refine-steps-val");

    // Toggle refinement steps visibility based on refine_gs checkbox
    const refineGsCheckbox = document.getElementById("refine-gs-input");
    const refineStepsGroup = document.getElementById("refine-steps-group");
    refineGsCheckbox.addEventListener("change", () => {
        if (refineGsCheckbox.checked) {
            refineStepsGroup.style.opacity = "1";
            refineStepsGroup.style.pointerEvents = "auto";
        } else {
            refineStepsGroup.style.opacity = "0.4";
            refineStepsGroup.style.pointerEvents = "none";
        }
    });

    // 5. Tab Switching Logic
    const tabButtons = document.querySelectorAll(".tab-btn");
    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            if (btn.disabled) return;
            
            // Remove active state
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
            
            // Add active state to target
            btn.classList.add("active");
            const tabId = btn.getAttribute("data-tab");
            document.getElementById(tabId).classList.add("active");
        });
    });

    // 6. Drag & Drop Upload Logic
    let selectedFile = null;

    dropzone.addEventListener("click", (e) => {
        if (e.target !== removeImgBtn && !removeImgBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    fileInput.addEventListener("change", (e) => {
        handleFiles(e.target.files);
    });

    // Drag events
    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        }, false);
    });

    dropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    removeImgBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        selectedFile = null;
        fileInput.value = "";
        imagePreview.src = "";
        dropzonePreview.style.display = "none";
        dropzoneEmpty.style.display = "block";
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith("image/")) {
                selectedFile = file;
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreview.src = e.target.result;
                    dropzoneEmpty.style.display = "none";
                    dropzonePreview.style.display = "flex";
                };
                reader.readAsDataURL(file);
            } else {
                alert(translations[currentLang].alert_select_image);
            }
        }
    }

    // 7. Submit & Pipeline Polling
    let pollInterval = null;

    generateBtn.addEventListener("click", () => {
        if (!selectedFile) {
            alert(translations[currentLang].alert_upload_image);
            return;
        }

        // Reset outputs
        modelViewer.style.display = "none";
        modelViewer.src = "";
        viewportPlaceholder.style.display = "flex";
        
        maskPreview.style.display = "none";
        maskPreview.src = "";
        maskPlaceholder.style.display = "flex";
        
        renderPreview.style.display = "none";
        renderPreview.src = "";
        renderPlaceholder.style.display = "flex";
        
        downloadsTabBtn.disabled = true;
        
        // Show status panel and logger
        statusBox.style.display = "block";
        logOutput.innerHTML = "";
        addLogLine(translations[currentLang].log_submitting, "cyan");
        
        // Gather params
        const formData = new FormData();
        formData.append("image", selectedFile);
        formData.append("seed", document.getElementById("seed-input").value);
        formData.append("ss_steps", document.getElementById("ss-steps-input").value);
        formData.append("ss_cfg", document.getElementById("ss-cfg-input").value);
        formData.append("slat_steps", document.getElementById("slat-steps-input").value);
        formData.append("slat_cfg", document.getElementById("slat-cfg-input").value);
        formData.append("refine_gs", refineGsCheckbox.checked);
        formData.append("refine_steps", document.getElementById("refine-steps-input").value);
        
        const upscaleFactor = document.querySelector('input[name="upscale-factor"]:checked').value;
        formData.append("upscale_factor", upscaleFactor);
        
        const upscaleDevice = document.querySelector('input[name="upscale-device"]:checked').value;
        formData.append("upscale_device", upscaleDevice);

        // Lock button
        generateBtn.disabled = true;
        generateBtn.textContent = translations[currentLang].btn_generating;
        
        // Send request
        fetch("/api/generate", {
            method: "POST",
            body: formData
        })
        .then(res => {
            if (!res.ok) throw new Error("API request failed.");
            return res.json();
        })
        .then(data => {
            const taskId = data.task_id;
            addLogLine(translations[currentLang].log_registered + taskId, "purple");
            startPolling(taskId);
        })
        .catch(err => {
            addLogLine(translations[currentLang].log_failed_submit + err.message, "red");
            resetGenerateBtn();
        });
    });

    function startPolling(taskId) {
        if (pollInterval) clearInterval(pollInterval);
        
        let lastStage = "";
        
        pollInterval = setInterval(() => {
            fetch(`/api/task/${taskId}`)
            .then(res => {
                if (!res.ok) throw new Error("Failed to poll.");
                return res.json();
            })
            .then(task => {
                // Update progress & status
                statusText.setAttribute("data-status", task.status);
                statusText.textContent = translations[currentLang]["status_" + task.status];
                statusPercent.textContent = `${task.progress}%`;
                progressBar.style.width = `${task.progress}%`;
                
                if (task.stage && task.stage !== lastStage) {
                    let logStage = task.stage;
                    if (currentLang === "en" && stageTranslations[task.stage]) {
                        logStage = stageTranslations[task.stage];
                    }
                    addLogLine(logStage, "cyan");
                    lastStage = task.stage;
                }
                
                if (task.status === "completed") {
                    clearInterval(pollInterval);
                    addLogLine(translations[currentLang].log_success, "green");
                    lastResults = task.result;
                    loadResults(task.result);
                    resetGenerateBtn();
                } else if (task.status === "failed") {
                    clearInterval(pollInterval);
                    addLogLine(translations[currentLang].log_failed + (task.error || translations[currentLang].unknown_error), "red");
                    resetGenerateBtn();
                }
            })
            .catch(err => {
                clearInterval(pollInterval);
                addLogLine(translations[currentLang].log_poll_error + err.message, "red");
                resetGenerateBtn();
            });
        }, 1000);
    }

    function loadResults(result) {
        // 1. Set 3D model viewport
        viewportPlaceholder.style.display = "none";
        modelViewer.src = result.glb_url;
        modelViewer.style.display = "block";
        
        // 2. Set mask and render previews
        maskPlaceholder.style.display = "none";
        maskPreview.src = result.mask_url;
        maskPreview.style.display = "block";
        
        renderPlaceholder.style.display = "none";
        renderPreview.src = result.render_url;
        renderPreview.style.display = "block";
        
        // 3. Populate downloads list
        downloadsTabBtn.disabled = false;
        
        const files = [
            { name: "forma_web.glb", type: translations[currentLang].file_glb_desc, icon: `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="file-svg"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>`, url: result.glb_url },
            { name: "forma_web_obj.zip", type: translations[currentLang].file_obj_desc, icon: `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="file-svg"><rect x="3" y="3" width="18" height="4" rx="1"></rect><path d="M4 7v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7"></path><path d="M10 12h4"></path></svg>`, url: result.obj_url },
            { name: "forma_web.ply", type: translations[currentLang].file_ply_desc, icon: `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="file-svg"><path d="M12 3l1.912 5.813a2 2 0 0 0 1.275 1.275L21 12l-5.813 1.912a2 2 0 0 0-1.275 1.275L12 21l-1.912-5.813a2 2 0 0 0-1.275-1.275L3 12l5.813-1.912a2 2 0 0 0 1.275-1.275L12 3z"></path></svg>`, url: result.ply_url },
            { name: "forma_web_metadata.json", type: translations[currentLang].file_json_desc, icon: `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="file-svg"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>`, url: result.meta_url }
        ];
        
        filesList.innerHTML = files.map(file => `
            <div class="file-item">
                <div class="file-info">
                    <span class="file-icon">${file.icon}</span>
                    <div>
                        <div class="file-name">${file.name}</div>
                        <div class="file-size">${file.type}</div>
                    </div>
                </div>
                <a href="${file.url}" download class="btn-download">${translations[currentLang].btn_download}</a>
            </div>
        `).join("");
    }

    function resetGenerateBtn() {
        generateBtn.disabled = false;
        generateBtn.textContent = translations[currentLang].generate_btn;
    }

    function addLogLine(text, colorClass) {
        const line = document.createElement("div");
        line.className = `log-line ${colorClass ? 'text-' + colorClass : ''}`;
        line.textContent = `> ${text}`;
        logOutput.appendChild(line);
        logOutput.scrollTop = logOutput.scrollHeight;
    }

    // Helper functions for Accordions
    function setupAccordion(headerId, contentId) {
        const header = document.getElementById(headerId);
        const parent = header.parentElement;
        header.addEventListener("click", () => {
            parent.classList.toggle("open");
            recalculateAccordionHeight(contentId);
        });
    }

    // Helper to dynamically calculate height
    function recalculateAccordionHeight(contentId) {
        const content = document.getElementById(contentId);
        const parent = content.parentElement;
        if (parent.classList.contains("open")) {
            content.style.maxHeight = content.scrollHeight + "px";
        } else {
            content.style.maxHeight = "0px";
        }
    }

    // Helper function for Slider values displaying dynamically
    function setupSliderBadge(sliderId, badgeId) {
        const slider = document.getElementById(sliderId);
        const badge = document.getElementById(badgeId);
        
        slider.addEventListener("input", () => {
            badge.textContent = slider.value;
        });
    }
});
