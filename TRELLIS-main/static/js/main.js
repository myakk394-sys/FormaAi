// JavaScript logic for FormaAI TRELLIS 3D Web UI

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const uploadZone = document.getElementById("uploadZone");
    const fileInput = document.getElementById("fileInput");
    const uploadContent = document.getElementById("uploadContent");
    const previewContainer = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const btnResetImage = document.getElementById("btnResetImage");
    
    const generationForm = document.getElementById("generationForm");
    const sessionIdInput = document.getElementById("sessionId");
    const seedInput = document.getElementById("seed");
    const randomizeSeedCheckbox = document.getElementById("randomizeSeed");
    const btnGenerate = document.getElementById("btnGenerate");
    
    const tabPreview = document.getElementById("tabPreview");
    const tab3dViewer = document.getElementById("tab3dViewer");
    const contentPreview = document.getElementById("contentPreview");
    const content3dViewer = document.getElementById("content3dViewer");
    
    const videoPlaceholder = document.getElementById("videoPlaceholder");
    const videoContainer = document.getElementById("videoContainer");
    const previewVideo = document.getElementById("previewVideo");
    
    const glbViewer = document.getElementById("glbViewer");
    const exportControls = document.getElementById("exportControls");
    const btnExtractGlb = document.getElementById("btnExtractGlb");
    const linkDownloadGlb = document.getElementById("linkDownloadGlb");
    const linkDownloadPly = document.getElementById("linkDownloadPly");
    
    const historyGallery = document.getElementById("historyGallery");
    const galleryEmpty = document.getElementById("galleryEmpty");
    const btnRefreshHistory = document.getElementById("btnRefreshHistory");
    
    // Accordions
    const accordions = document.querySelectorAll(".accordion-title");
    accordions.forEach(acc => {
        acc.addEventListener("click", () => {
            acc.parentElement.classList.toggle("open");
        });
    });

    // Seed Randomize behavior
    randomizeSeedCheckbox.addEventListener("change", (e) => {
        seedInput.disabled = e.target.checked;
    });

    // Tabs switching
    tabPreview.addEventListener("click", () => {
        tabPreview.classList.add("active");
        tab3dViewer.classList.remove("active");
        contentPreview.style.display = "block";
        content3dViewer.style.display = "none";
    });

    tab3dViewer.addEventListener("click", () => {
        tab3dViewer.classList.add("active");
        tabPreview.classList.remove("active");
        content3dViewer.style.display = "block";
        contentPreview.style.display = "none";
    });

    // Upload & Preprocess logic
    uploadZone.addEventListener("click", (e) => {
        if (e.target !== btnResetImage && !btnResetImage.contains(e.target)) {
            fileInput.click();
        }
    });

    uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadZone.classList.add("dragover");
    });

    uploadZone.addEventListener("dragleave", () => {
        uploadZone.classList.remove("dragover");
    });

    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.classList.remove("dragover");
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleImageUpload(files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        const files = e.target.files;
        if (files.length > 0) {
            handleImageUpload(files[0]);
        }
    });

    btnResetImage.addEventListener("click", (e) => {
        e.stopPropagation();
        resetImageUpload();
    });

    function showToast(message, type = "info") {
        const toastContainer = document.getElementById("toastContainer");
        const toast = document.createElement("div");
        toast.className = `toast ${type}`;
        
        let icon = "fa-info-circle";
        if (type === "success") icon = "fa-check-circle";
        if (type === "warning") icon = "fa-exclamation-triangle";
        if (type === "error") icon = "fa-exclamation-circle";
        
        toast.innerHTML = `
            <i class="fa-solid ${icon}"></i>
            <span>${message}</span>
        `;
        toastContainer.appendChild(toast);
        
        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateX(100%)";
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    function resetImageUpload() {
        fileInput.value = "";
        previewContainer.style.display = "none";
        imagePreview.src = "";
        uploadContent.style.display = "flex";
        sessionIdInput.value = "";
        btnGenerate.disabled = true;
    }

    async function handleImageUpload(file) {
        // Show loading state
        uploadContent.innerHTML = `
            <i class="fa-solid fa-circle-notch fa-spin upload-icon" style="color: var(--primary);"></i>
            <p class="upload-title">Обработка изображения...</p>
            <p class="upload-subtitle">Удаление фона и кадрирование</p>
        `;
        
        const formData = new FormData();
        formData.append("file", file);
        
        try {
            const response = await fetch("/api/preprocess", {
                method: "POST",
                body: formData
            });
            const data = await response.json();
            
            if (!response.ok) throw new Error(data.detail || "Error during preprocessing");
            
            // Show preview of processed image
            imagePreview.src = data.processedUrl;
            uploadContent.style.display = "none";
            previewContainer.style.display = "block";
            
            // Set session id
            sessionIdInput.value = data.id;
            btnGenerate.disabled = false;
            
            showToast("Изображение обработано успешно!", "success");
        } catch (error) {
            console.error(error);
            showToast("Ошибка при обработке изображения: " + error.message, "error");
            resetImageUpload();
            uploadContent.innerHTML = `
                <i class="fa-solid fa-images upload-icon"></i>
                <p class="upload-title">Перетащите изображение сюда</p>
                <p class="upload-subtitle">или нажмите для выбора файла</p>
                <span class="upload-limit">Поддержка PNG, JPG (будет автоматически удален фон)</span>
            `;
        }
    }

    // Generation Submit
    generationForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const formData = new FormData(generationForm);
        
        // UI loading state
        const btnText = btnGenerate.querySelector(".btn-text");
        const spinner = btnGenerate.querySelector(".spinner");
        btnGenerate.disabled = true;
        btnText.style.display = "none";
        spinner.style.display = "inline-flex";
        
        // Reset output elements
        videoPlaceholder.style.display = "flex";
        videoPlaceholder.innerHTML = `
            <div class="placeholder-icon-wrap">
                <i class="fa-solid fa-circle-notch fa-spin placeholder-icon" style="color: var(--primary);"></i>
            </div>
            <h3>Идет 3D генерация...</h3>
            <p>Это может занять от 15 до 45 секунд в зависимости от загрузки GPU</p>
        `;
        videoContainer.style.display = "none";
        tab3dViewer.disabled = true;
        exportControls.style.display = "none";
        
        showToast("Запущена генерация 3D модели...", "info");
        
        try {
            const response = await fetch("/api/generate", {
                method: "POST",
                body: formData
            });
            const data = await response.json();
            
            if (!response.ok) throw new Error(data.detail || "Error during generation");
            
            // Show video output
            videoPlaceholder.style.display = "none";
            videoContainer.style.display = "block";
            previewVideo.src = data.videoUrl;
            previewVideo.play();
            
            // Setup download ply
            linkDownloadPly.href = data.plyUrl;
            linkDownloadPly.style.display = "inline-flex";
            
            // Enable GLB extraction controls
            exportControls.style.display = "block";
            btnExtractGlb.disabled = false;
            const btnExtractText = btnExtractGlb.querySelector(".btn-text");
            const btnExtractSpinner = btnExtractGlb.querySelector(".spinner");
            btnExtractText.style.display = "inline-flex";
            btnExtractSpinner.style.display = "none";
            
            // Disable GLB download button until extracted
            linkDownloadGlb.classList.add("disabled");
            linkDownloadGlb.href = "#";
            
            showToast("Генерация завершена успешно!", "success");
            loadHistoryGallery();
        } catch (error) {
            console.error(error);
            showToast("Ошибка при генерации 3D: " + error.message, "error");
            videoPlaceholder.innerHTML = `
                <div class="placeholder-icon-wrap" style="background: rgba(239, 68, 68, 0.1); border-color: rgba(239, 68, 68, 0.2)">
                    <i class="fa-solid fa-circle-exclamation placeholder-icon" style="color: var(--danger);"></i>
                </div>
                <h3>Ошибка генерации</h3>
                <p>${error.message}</p>
            `;
        } finally {
            btnText.style.display = "inline-flex";
            spinner.style.display = "none";
            btnGenerate.disabled = false;
        }
    });

    // GLB Extraction
    btnExtractGlb.addEventListener("click", async () => {
        const id = sessionIdInput.value;
        const meshSimplify = document.getElementById("meshSimplify").value;
        const textureSize = document.getElementById("textureSize").value;
        
        const formData = new FormData();
        formData.append("id", id);
        formData.append("mesh_simplify", meshSimplify);
        formData.append("texture_size", textureSize);
        
        // Loading state
        btnExtractGlb.disabled = true;
        const btnText = btnExtractGlb.querySelector(".btn-text");
        const spinner = btnExtractGlb.querySelector(".spinner");
        btnText.style.display = "none";
        spinner.style.display = "inline-flex";
        
        showToast("Запущено извлечение полигональной сетки и запекание текстур...", "info");
        
        try {
            const response = await fetch("/api/extract_glb", {
                method: "POST",
                body: formData
            });
            const data = await response.json();
            
            if (!response.ok) throw new Error(data.detail || "Error during GLB extraction");
            
            // Set GLB viewer source
            glbViewer.src = data.glbUrl;
            
            // Enable 3D Tab and switch
            tab3dViewer.disabled = false;
            tab3dViewer.click(); // Switch to 3D tab
            
            // Enable GLB download link
            linkDownloadGlb.classList.remove("disabled");
            linkDownloadGlb.href = data.glbUrl;
            
            showToast("GLB модель успешно извлечена!", "success");
            loadHistoryGallery();
        } catch (error) {
            console.error(error);
            showToast("Ошибка при извлечении GLB: " + error.message, "error");
        } finally {
            btnText.style.display = "inline-flex";
            spinner.style.display = "none";
            btnExtractGlb.disabled = false;
        }
    });

    // Load History list
    async function loadHistoryGallery() {
        try {
            const response = await fetch("/api/history");
            const history = await response.json();
            
            // Clear gallery
            // Keep empty state element, but hide/show appropriately
            const cards = historyGallery.querySelectorAll(".history-card");
            cards.forEach(card => card.remove());
            
            if (history.length === 0) {
                galleryEmpty.style.display = "flex";
                return;
            }
            
            galleryEmpty.style.display = "none";
            
            history.forEach(item => {
                const card = document.createElement("div");
                card.className = "history-card";
                card.id = `card-${item.id}`;
                
                card.innerHTML = `
                    <div class="card-media">
                        <video src="${item.videoUrl}" loop muted playsinline autoplay></video>
                        <span class="card-badge">Seed: ${item.seed}</span>
                        <div class="card-delete" title="Удалить"><i class="fa-solid fa-trash"></i></div>
                    </div>
                    <div class="card-info">
                        <div class="card-title">Генерация #${item.id.slice(0, 8)}</div>
                        <div class="card-meta">Шаги: ${item.ss_sampling_steps}/${item.slat_sampling_steps}</div>
                        <div class="card-actions">
                            <button class="btn btn-secondary btn-sm btn-load-model" data-id="${item.id}" data-video="${item.videoUrl}" data-ply="${item.plyUrl}" data-glb="${item.glbUrl || ''}">
                                <i class="fa-solid fa-folder-open"></i> Загрузить
                            </button>
                        </div>
                    </div>
                `;
                
                // Bind delete event
                const btnDelete = card.querySelector(".card-delete");
                btnDelete.addEventListener("click", (e) => {
                    e.stopPropagation();
                    deleteHistoryItem(item.id);
                });
                
                // Bind load event
                const btnLoad = card.querySelector(".btn-load-model");
                btnLoad.addEventListener("click", () => {
                    loadModelToWorkspace(item);
                });
                
                historyGallery.appendChild(card);
            });
        } catch (error) {
            console.error("Error loading history:", error);
        }
    }

    async function deleteHistoryItem(id) {
        if (!confirm("Вы уверены, что хотите удалить эту генерацию?")) return;
        
        const formData = new FormData();
        formData.append("id", id);
        
        try {
            const response = await fetch("/api/delete", {
                method: "POST",
                body: formData
            });
            if (response.ok) {
                showToast("Генерация удалена", "info");
                document.getElementById(`card-${id}`).remove();
                
                // Check if empty
                if (historyGallery.querySelectorAll(".history-card").length === 0) {
                    galleryEmpty.style.display = "flex";
                }
            } else {
                showToast("Не удалось удалить элемент", "error");
            }
        } catch (error) {
            showToast("Ошибка при удалении: " + error.message, "error");
        }
    }

    function loadModelToWorkspace(item) {
        // Load image to preview
        imagePreview.src = item.processedUrl;
        uploadContent.style.display = "none";
        previewContainer.style.display = "block";
        
        sessionIdInput.value = item.id;
        btnGenerate.disabled = false;
        
        // Show video preview
        videoPlaceholder.style.display = "none";
        videoContainer.style.display = "block";
        previewVideo.src = item.videoUrl;
        previewVideo.play();
        
        // Configure downloads
        linkDownloadPly.href = item.plyUrl;
        linkDownloadPly.style.display = "inline-flex";
        
        exportControls.style.display = "block";
        btnExtractGlb.disabled = false;
        
        // Load GLB if already extracted
        if (item.glbUrl) {
            glbViewer.src = item.glbUrl;
            tab3dViewer.disabled = false;
            tab3dViewer.click(); // Switch to 3D tab
            
            linkDownloadGlb.classList.remove("disabled");
            linkDownloadGlb.href = item.glbUrl;
        } else {
            // Clean glb viewer source
            glbViewer.removeAttribute("src");
            tab3dViewer.disabled = true;
            tabPreview.click(); // Go back to video preview
            
            linkDownloadGlb.classList.add("disabled");
            linkDownloadGlb.href = "#";
        }
        
        showToast(`Загружена генерация #${item.id.slice(0, 8)}`, "info");
    }

    btnRefreshHistory.addEventListener("click", loadHistoryGallery);

    // Initial history load
    loadHistoryGallery();
});
