import { state } from '@angular/animations';
import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { ToastrService } from 'ngx-toastr';
import { ToastServiceService } from '../toast-service.service';

interface FaceDetectionResponse {
  detected_faces_count: number;
  processed_image_path: string;
}

@Component({
  selector: 'app-home-page',
  templateUrl: './home-page.component.html',
  styleUrls: ['./home-page.component.scss']
})
export class HomePageComponent {
  isDragging = false;
  selectedFile: File | null = null;
  previewUrl: string | ArrayBuffer | null = null;

  constructor(private http: HttpClient
            ,private router: Router,
          private toast : ToastServiceService) {
    console.log('HomePageComponent constructed');
  }

  onDragEnter(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging = true;
  }

  onDragLeave(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging = false;
  }

  onDragOver(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging = true;
  }

  onDrop(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging = false;
    const files = event.dataTransfer?.files;
    if (files && files.length > 0) {
      this.handleFileInput(files[0]);
    }
  }

  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.handleFileInput(input.files[0]);
    }
  }

  handleFileInput(file: File) {
    this.selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      this.previewUrl = e.target?.result as string;
    };
    reader.readAsDataURL(file);
  }

  

  onSubmit(event: Event) {
    event.preventDefault();
    console.log("Form submitted");
    if (this.selectedFile) {
      const formData = new FormData();
      formData.append('image', this.selectedFile, this.selectedFile.name);

      this.http.post<FaceDetectionResponse>('http://127.0.0.1:5000/detect_faces', formData)
        .subscribe(
          (response) => {
            console.log('API Response:', response);
            localStorage.setItem("result", response.detected_faces_count.toString());
            localStorage.setItem("image", response.processed_image_path); 
            this.router.navigate(['/result']);
            // this.toast.showSuccess("Image uploaded successfully.");
          },
          (error) => {
            console.error('API Error:', error.error.error);
            this.toast.showError(error.error.error);
          }
        );
    } else {
      console.error('No file selected');
    }
  }
}